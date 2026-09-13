"""Esleme katmani birim testleri.

Bunlar Blender gerektirmez ve hizli kosar. Rig tarafinda bir sey bozuldugunda
once buraya bakilir: eger burasi yesilse hata tespit/eksen/olcek katmanindadir,
esleme katmaninda degildir.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.calibration import Profile, build_neutral, mirror_blendshapes  # noqa: E402
from core.filters import OneEuroFilter  # noqa: E402
from core.mapping import Mapping, split_axis_key  # noqa: E402
from core.take import read_take  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "take_5s.jsonl"


@pytest.fixture(scope="module")
def mapping() -> Mapping:
    return Mapping.load()


# --------------------------------------------------------------------------
# tablo butunlugu
# --------------------------------------------------------------------------


def test_tablo_mediapipe_isimleriyle_anahtarlanmis(mapping):
    """Anahtarlar camelCase olmali. PascalCase sizarsa detector ciktisi
    tabloyla eslesmez ve her sey sessizce sifir kalir."""
    for name in mapping.table:
        assert name[0].islower(), f"{name} camelCase degil"


def test_jaw_open_tek_eksene_dusuyor(mapping):
    entries = mapping.table["jawOpen"]
    assert len(entries) == 1
    assert entries[0]["gui"] == "CTRL_C_jaw"
    assert entries[0]["axis"] == "y"
    assert entries[0]["weight"] == pytest.approx(1.0, abs=1e-5)


def test_blink_ve_wide_ayni_eksenin_zit_yarilari(mapping):
    """DNA'da eyeWidenL, eyeBlinkL ile ayni GUI slider'inin negatif
    yarisindan suruluyor. Isaretler ters degilse goz acma kirpma gibi
    davranir -- bu testin yakaladigi hata sinifi tam olarak bu."""
    blink = mapping.table["eyeBlinkLeft"][0]
    wide = mapping.table["eyeWideLeft"][0]
    assert blink["gui"] == wide["gui"] == "CTRL_L_eye_blink"
    assert blink["axis"] == wide["axis"] == "y"
    assert blink["weight"] > 0 > wide["weight"]


def test_her_eksen_icin_limit_tanimli(mapping):
    for axis_key in mapping.axes:
        assert axis_key in mapping.limits, f"{axis_key} icin guiLimits yok"


def test_gui_isimleri_dna_kemik_adi_formatinda(mapping):
    for bone in mapping.bones:
        assert bone.startswith("CTRL_"), bone


# --------------------------------------------------------------------------
# uygulama mantigi
# --------------------------------------------------------------------------


def test_bos_girdi_bos_cikti(mapping):
    assert mapping.apply({}) == {}


def test_neutral_kanali_yok_sayiliyor(mapping):
    assert mapping.apply({"_neutral": 1.0}) == {}


def test_bilinmeyen_kanal_sessizce_yutulmuyor(mapping):
    assert mapping.unknown_shapes({"jawOpen": 1.0, "uydurmaKanal": 0.5}) == ["uydurmaKanal"]


def test_blink_ve_wide_toplanip_birbirini_goturuyor(mapping):
    """max alsaydik bu test gecmezdi: ikisi de 0.5 iken sonuc 0.5 cikardi,
    dogrusu 0."""
    result = mapping.apply({"eyeBlinkLeft": 0.5, "eyeWideLeft": 0.5})
    assert result["CTRL_L_eye_blink.y"] == pytest.approx(0.0, abs=1e-5)


def test_deger_limit_disina_tasmiyor(mapping):
    # ayni eksene dusen birden fazla kanali sonuna kadar acinca
    result = mapping.apply({name: 1.0 for name in mapping.table})
    for axis_key, value in result.items():
        limit = mapping.limits[axis_key]
        assert limit["min"] - 1e-6 <= value <= limit["max"] + 1e-6, f"{axis_key}={value}"


def test_only_filtresi_tek_kanali_izole_ediyor(mapping):
    """Plan geregi once tek kanalla dogrulama yapiliyor; bu yolun
    calistigini garanti eder."""
    result = mapping.apply({"jawOpen": 0.5, "mouthSmileLeft": 1.0}, only={"jawOpen"})
    assert list(result) == ["CTRL_C_jaw.y"]


def test_split_axis_key():
    assert split_axis_key("CTRL_C_jaw.y") == ("CTRL_C_jaw", 1)
    assert split_axis_key("CTRL_L_eye_blink.x") == ("CTRL_L_eye_blink", 0)


# --------------------------------------------------------------------------
# kalibrasyon ve filtre
# --------------------------------------------------------------------------


def test_notr_offset_cikariliyor():
    profile = Profile(neutral={"jawOpen": 0.2})
    # notr degerin kendisi sifira inmeli
    assert profile.apply({"jawOpen": 0.2})["jawOpen"] == pytest.approx(0.0)
    # tam acik yine 1.0 kalmali, aralik yeniden olcekleniyor
    assert profile.apply({"jawOpen": 1.0})["jawOpen"] == pytest.approx(1.0)


def test_deadzone_ve_gain():
    profile = Profile(channels={"jawOpen": __import__("core.calibration", fromlist=["ChannelProfile"]).ChannelProfile(gain=2.0, deadzone=0.1)})
    assert profile.apply({"jawOpen": 0.05})["jawOpen"] == 0.0
    assert profile.apply({"jawOpen": 0.4})["jawOpen"] == pytest.approx(0.8)


def test_mirror_sol_sag_takasi():
    assert mirror_blendshapes({"eyeBlinkLeft": 1.0}) == {"eyeBlinkRight": 1.0}
    assert mirror_blendshapes({"jawOpen": 1.0}) == {"jawOpen": 1.0}


def test_one_euro_sabit_sinyali_bozmuyor():
    filt = OneEuroFilter()
    for index in range(50):
        value = filt(0.5, index / 30.0)
    assert value == pytest.approx(0.5, abs=1e-3)


def test_one_euro_gurultuyu_azaltiyor():
    import random

    random.seed(0)
    filt = OneEuroFilter(min_cutoff=0.5, beta=0.0)
    noisy, filtered = [], []
    for index in range(200):
        raw = 0.5 + random.uniform(-0.1, 0.1)
        noisy.append(raw)
        filtered.append(filt(raw, index / 30.0))
    spread = lambda xs: max(xs[50:]) - min(xs[50:])  # noqa: E731
    assert spread(filtered) < spread(noisy) * 0.6


# --------------------------------------------------------------------------
# fixture uzerinden uctan uca (Blender'siz)
# --------------------------------------------------------------------------


@pytest.mark.skipif(not FIXTURE.exists(), reason="once scripts/03_make_fixture.py calistir")
def test_fixture_bastan_sona_isleniyor(mapping):
    frames = list(read_take(FIXTURE))
    assert len(frames) == 150, "5 sn @ 30fps"

    profile = Profile(neutral=build_neutral(iter(frames), sample_count=30))

    for frame in frames:
        assert mapping.unknown_shapes(frame.bs) == []
        values = mapping.apply(profile.apply(frame.bs))
        assert values, f"kare {frame.frame} bos cikti"
        for axis_key, value in values.items():
            limit = mapping.limits[axis_key]
            assert limit["min"] - 1e-6 <= value <= limit["max"] + 1e-6


@pytest.mark.skipif(not FIXTURE.exists(), reason="once scripts/03_make_fixture.py calistir")
def test_fixture_kare_numaralari_artiyor():
    frames = list(read_take(FIXTURE))
    assert [f.frame for f in frames] == list(range(1, len(frames) + 1))


def test_bilinen_bosluk_mekanizmasi():
    """Bilinen bosluklar uyari uretmemeli, gercek isim kaymasi uretmeli.

    Tabloda su an bosluk yok (51/51), o yuzden mekanizma sentetik belge ile
    sinaniyor -- kapsama duzeldi diye test kaybolmasin."""
    from core.mapping import Mapping

    belge = {
        "arkit": {"jawOpen": [{"gui": "CTRL_C_jaw", "axis": "y", "weight": 1.0}]},
        "guiLimits": {"CTRL_C_jaw.y": {"min": 0.0, "max": 1.0}},
        "knownUnmapped": ["mouthClose"],
    }
    m = Mapping(belge)
    assert m.unknown_shapes({"jawOpen": 1.0, "mouthClose": 0.3}) == []
    assert m.unknown_shapes({"mouthClose": 0.3, "jawOpenn": 1.0}) == ["jawOpenn"]


def test_mouth_close_bagli(mapping):
    """Dudak kapanmasi konusmanin dogal gorunmesinin sarti: 'm', 'b', 'p'
    seslerinde cene acikken dudaklar kapanmali. Bu esleme Epic'in
    asset'inden turemiyor, elle yazildi -- kaybolursa fark edilmeli."""
    entries = mapping.table.get("mouthClose")
    assert entries, "mouthClose eslemesi yok"
    hedefler = {e["gui"] for e in entries}
    assert hedefler == {
        "CTRL_L_mouth_lipsTogetherU",
        "CTRL_R_mouth_lipsTogetherU",
        "CTRL_L_mouth_lipsTogetherD",
        "CTRL_R_mouth_lipsTogetherD",
    }
    assert all(e["weight"] > 0 for e in entries)


def test_tum_mediapipe_kanallari_kapsandi(mapping):
    from tests.scripts_shim import MEDIAPIPE_BLENDSHAPES

    eksik = [n for n in MEDIAPIPE_BLENDSHAPES if n not in mapping.table]
    assert eksik == [], f"kapsanmayan kanal: {eksik}"


# --------------------------------------------------------------------------
# kanal gruplari
# --------------------------------------------------------------------------


def test_grup_ataması():
    from core.calibration import group_of

    assert group_of("jawOpen") == "jaw"
    assert group_of("mouthClose") == "lipClose"
    assert group_of("mouthSmileLeft") == "lips"
    assert group_of("eyeBlinkLeft") == "eyes"
    assert group_of("browInnerUp") == "brows"
    assert group_of("cheekPuff") == "cheeksNose"


def test_grup_kazanci_uygulaniyor():
    from core.calibration import ChannelProfile, Profile

    profile = Profile(groups={"lips": ChannelProfile(gain=0.5), "jaw": ChannelProfile(gain=2.0)})
    out = profile.apply({"mouthSmileLeft": 0.4, "jawOpen": 0.3, "browInnerUp": 0.6})
    assert out["mouthSmileLeft"] == pytest.approx(0.2)
    assert out["jawOpen"] == pytest.approx(0.6)
    assert out["browInnerUp"] == pytest.approx(0.6), "grubu olmayan kanal degismemeli"


def test_kanala_ozel_ayar_grubu_eziyor():
    from core.calibration import ChannelProfile, Profile

    profile = Profile(
        groups={"lips": ChannelProfile(gain=0.5)},
        channels={"mouthSmileLeft": ChannelProfile(gain=3.0)},
    )
    out = profile.apply({"mouthSmileLeft": 0.2, "mouthFrownLeft": 0.2})
    assert out["mouthSmileLeft"] == pytest.approx(0.6)
    assert out["mouthFrownLeft"] == pytest.approx(0.1)


def test_agiz_kanallari_daha_az_yumusatiliyor():
    """Konusma hizli; agiza kaslarla ayni filtreyi uygularsan dudaklar
    lapa oluyor ve kelimeler okunmuyor."""
    from core.filters import ChannelFilters

    filters = ChannelFilters()
    agiz = filters._get("mouthFunnel")
    kas = filters._get("browInnerUp")
    assert agiz.min_cutoff > kas.min_cutoff


# --------------------------------------------------------------------------
# kafa pozu
# --------------------------------------------------------------------------


def test_kafa_notr_referansa_gore_bagil():
    """Ayni yonelim surekli gelirse kafa DONMEMELI. Mutlak deger yazsaydik
    karakterin kafasi kameraya gore sabitlenirdi."""
    from core.head import HeadTracker

    tracker = HeadTracker()
    egik = [0.0, 0.3826834, 0.0, 0.9238795]  # 45 derece
    for _ in range(40):
        w, x, y, z = tracker.feed(egik)
    assert (w, x, y, z) == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-4)


def test_kafa_notrden_sapinca_doner():
    from core.head import HeadTracker

    tracker = HeadTracker(sample_count=1)
    tracker.feed([0.0, 0.0, 0.0, 1.0])
    w, x, y, z = tracker.feed([0.0, 0.3826834, 0.0, 0.9238795])
    assert abs(w) < 0.99, "sapma varken donme uretilmedi"


def test_yaw_secilen_eksene_gidiyor():
    """Belirtiden ayara: 'saga cevirince asagi bakiyor' demek yaw'in yanlis
    eksende oldugu demek. Bu test yaw'in SADECE secilen eksende donme
    urettigini garanti eder."""
    from core.head import HeadTracker

    saga_donmus = [0.0, 0.3826834, 0.0, 0.9238795]  # Y ekseni = yaw

    for axis, index in (("X", 1), ("Y", 2), ("Z", 3)):
        tracker = HeadTracker(
            mapping={"yaw": (axis, 1.0), "pitch": ("", 1.0), "roll": ("", 1.0)},
            sample_count=1,
        )
        tracker.feed([0.0, 0.0, 0.0, 1.0])
        result = tracker.feed(saga_donmus)
        # blender sirasi (w, x, y, z) -> secilen eksen disi bilesenler sifir
        for other in (1, 2, 3):
            if other != index:
                assert result[other] == pytest.approx(0.0, abs=1e-6), f"{axis}: {other} sizdi"
        assert abs(result[index]) > 1e-3, f"{axis} ekseninde donme yok"


def test_yaw_ters_isaret():
    from core.head import HeadTracker

    saga_donmus = [0.0, 0.3826834, 0.0, 0.9238795]
    duz = {"pitch": ("", 1.0), "roll": ("", 1.0)}

    normal = HeadTracker(mapping={"yaw": ("Z", 1.0), **duz}, sample_count=1)
    normal.feed([0.0, 0.0, 0.0, 1.0])
    a = normal.feed(saga_donmus)

    ters = HeadTracker(mapping={"yaw": ("Z", -1.0), **duz}, sample_count=1)
    ters.feed([0.0, 0.0, 0.0, 1.0])
    b = ters.feed(saga_donmus)

    assert a[3] == pytest.approx(-b[3], abs=1e-6)


def test_kapali_eksen_donme_uretmiyor():
    from core.head import HeadTracker

    tracker = HeadTracker(
        mapping={"yaw": ("", 1.0), "pitch": ("", 1.0), "roll": ("", 1.0)}, sample_count=1
    )
    tracker.feed([0.0, 0.0, 0.0, 1.0])
    assert tracker.feed([0.0, 0.3826834, 0.0, 0.9238795]) == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_ayristirma_yaw_pitch_roll_ayiriyor():
    import math

    from core.head import decompose, quaternion_from_axis

    pitch, yaw, roll = decompose(quaternion_from_axis(1, math.radians(30)))  # Y = yaw
    assert yaw == pytest.approx(math.radians(30), abs=1e-4)
    assert abs(pitch) < 1e-6 and abs(roll) < 1e-6

    pitch, yaw, roll = decompose(quaternion_from_axis(0, math.radians(20)))  # X = pitch
    assert pitch == pytest.approx(math.radians(20), abs=1e-4)
    assert abs(yaw) < 1e-6 and abs(roll) < 1e-6


def test_kafa_etkisi_sifirken_donme_yok():
    from core.head import HeadTracker

    tracker = HeadTracker(influence=0.0, sample_count=1)
    tracker.feed([0.0, 0.0, 0.0, 1.0])
    assert tracker.feed([0.0, 0.3826834, 0.0, 0.9238795]) == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_kafa_blender_quaternion_sirasi():
    from core.head import to_blender_quaternion

    assert to_blender_quaternion([1.0, 2.0, 3.0, 4.0]) == (4.0, 1.0, 2.0, 3.0)


def test_birlestirme_birim_quaternion_uretiyor():
    import math

    from core.head import AXES, compose

    for axis in AXES:
        q = compose(
            math.radians(10),
            math.radians(20),
            math.radians(5),
            {"yaw": (axis, 1.0), "pitch": ("X", -1.0), "roll": ("Y", 1.0)},
        )
        assert sum(c * c for c in q) ** 0.5 == pytest.approx(1.0, abs=1e-6), axis


def test_kafa_verisi_yoksa_none():
    from core.head import HeadTracker

    tracker = HeadTracker()
    assert tracker.feed(None) is None
    assert tracker.feed([]) is None
