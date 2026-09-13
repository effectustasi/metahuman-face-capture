"""ARKit'in kanali olmayan kontrollerin turetilmesi.

Yonler DNA'dan dogrulandi; bu testler o yonlerin ters cevrilmesini yakalar.
Isaret hatasi burada sessizce gecerse kapaklar ters yone gider ve sebebini
bulmak zor olur.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.calibration import ChannelProfile, Profile, RangeLearner  # noqa: E402
from core.realism import EYE_SIDES, MicroSaccades, RealismLayer, merge  # noqa: E402


# --------------------------------------------------------------------------
# kapak bakisi takip eder
# --------------------------------------------------------------------------


def test_yukari_bakinca_ust_kapak_kalkar():
    """DNA'da eyelidU'nun NEGATIFI kapagi kaldiriyor. Yukari bakis (+)
    negatif eyelidU uretmeli -- isaret tersse goz yukari bakarken kapak
    inip goz kapanmis gorunur."""
    layer = RealismLayer(saccades=0.0)
    extra = layer.extra_axes({"CTRL_L_eye.y": 0.8})
    assert extra["CTRL_L_eye_eyelidU.y"] < 0


def test_asagi_bakinca_ust_kapak_iner():
    layer = RealismLayer(saccades=0.0)
    extra = layer.extra_axes({"CTRL_L_eye.y": -0.8})
    assert extra["CTRL_L_eye_eyelidU.y"] > 0


def test_duz_bakista_kapak_notr():
    layer = RealismLayer(saccades=0.0)
    extra = layer.extra_axes({"CTRL_L_eye.y": 0.0})
    assert extra["CTRL_L_eye_eyelidU.y"] == pytest.approx(0.0)


def test_alt_kapak_ust_kapagin_TERSI_yonde():
    """Ust kapak ve alt kapak ayni bakista zit isaret almali (DNA'da
    eyelidU'nun negatifi kapagi kaldirirken eyelidD'nin negatifi indiriyor)."""
    layer = RealismLayer(saccades=0.0)
    extra = layer.extra_axes({"CTRL_L_eye.y": 0.5})
    assert extra["CTRL_L_eye_eyelidU.y"] * extra["CTRL_L_eye_eyelidD.y"] < 0


def test_alt_kapak_ustten_daha_az_hareket_eder():
    layer = RealismLayer(saccades=0.0)
    extra = layer.extra_axes({"CTRL_L_eye.y": 0.9})
    assert abs(extra["CTRL_L_eye_eyelidD.y"]) < abs(extra["CTRL_L_eye_eyelidU.y"])


def test_goz_kapaliyken_takip_soneriyor():
    """Kirpma sirasinda kapak takibi kapagi geri acmamali."""
    layer = RealismLayer(saccades=0.0)
    acik = layer.extra_axes({"CTRL_L_eye.y": 0.8, "CTRL_L_eye_blink.y": 0.0})
    kapali = layer.extra_axes({"CTRL_L_eye.y": 0.8, "CTRL_L_eye_blink.y": 1.0})
    assert abs(kapali["CTRL_L_eye_eyelidU.y"]) < abs(acik["CTRL_L_eye_eyelidU.y"])
    assert kapali["CTRL_L_eye_eyelidU.y"] == pytest.approx(0.0)


def test_takip_kapatilabiliyor():
    layer = RealismLayer(eyelid_follow=0.0, lower_lid_follow=0.0, saccades=0.0)
    extra = layer.extra_axes({"CTRL_L_eye.y": 0.8})
    assert "CTRL_L_eye_eyelidU.y" not in extra


def test_iki_goz_de_suruluyor():
    layer = RealismLayer(saccades=0.0)
    extra = layer.extra_axes({"CTRL_L_eye.y": 0.5, "CTRL_R_eye.y": 0.5})
    for side in EYE_SIDES:
        assert side["upper"] in extra


# --------------------------------------------------------------------------
# kapak baskisi
# --------------------------------------------------------------------------


def test_kisilma_kapak_baskisi_uretiyor():
    layer = RealismLayer(saccades=0.0)
    extra = layer.extra_axes({"CTRL_L_eye_squintInner.y": 0.6})
    assert extra["CTRL_L_eye_lidPress.y"] > 0


def test_kisilma_yoksa_baski_yok():
    layer = RealismLayer(saccades=0.0)
    assert "CTRL_L_eye_lidPress.y" not in layer.extra_axes({"CTRL_L_eye.y": 0.3})


# --------------------------------------------------------------------------
# mikro-sakkadlar
# --------------------------------------------------------------------------


def test_sakkad_bakisi_oynatiyor():
    layer = RealismLayer(seed=7)
    hareket = set()
    for step in range(60):
        extra = layer.extra_axes({"CTRL_L_eye.x": 0.0, "CTRL_L_eye.y": 0.0}, timestamp=step / 30)
        hareket.add(round(extra["CTRL_L_eye.x"], 6))
    assert len(hareket) > 3, "sakkad hic hareket uretmedi"


def test_sakkad_kucuk_kaliyor():
    layer = RealismLayer(seed=7, saccade_amplitude=0.02)
    for step in range(300):
        extra = layer.extra_axes({"CTRL_L_eye.x": 0.0, "CTRL_L_eye.y": 0.0}, timestamp=step / 30)
        assert abs(extra["CTRL_L_eye.x"]) < 0.15, "sakkad gozu firlatiyor"


def test_sakkad_mevcut_bakisin_USTUNE_biniyor():
    layer = RealismLayer(seed=3, saccade_amplitude=0.01)
    extra = layer.extra_axes({"CTRL_L_eye.x": 0.5, "CTRL_L_eye.y": 0.0}, timestamp=0.0)
    assert abs(extra["CTRL_L_eye.x"] - 0.5) < 0.1, "sakkad bakisi eziyor"


def test_sakkad_kapatilabiliyor():
    layer = RealismLayer(saccades=0.0)
    extra = layer.extra_axes({"CTRL_L_eye.x": 0.4}, timestamp=1.0)
    assert "CTRL_L_eye.x" not in extra


def test_sakkad_deterministik():
    a = MicroSaccades(seed=42)
    b = MicroSaccades(seed=42)
    for step in range(20):
        assert a.offset(step / 30) == b.offset(step / 30)


# --------------------------------------------------------------------------
# birlestirme
# --------------------------------------------------------------------------


def test_merge_limitleri_uyguluyor():
    limits = {"CTRL_L_eye_eyelidU.y": {"min": -1.0, "max": 1.0}}
    result = merge({}, {"CTRL_L_eye_eyelidU.y": -5.0}, limits)
    assert result["CTRL_L_eye_eyelidU.y"] == -1.0


def test_merge_dokunulmayani_koruyor():
    result = merge({"CTRL_C_jaw.y": 0.7}, {"CTRL_L_eye_eyelidU.y": -0.3})
    assert result["CTRL_C_jaw.y"] == 0.7


# --------------------------------------------------------------------------
# kanal araligi
# --------------------------------------------------------------------------


def test_input_max_tepeyi_bire_cekiyor():
    """Gercek olcum: eyeBlink 0.917'de doyuyor, yani goz tam kapanmiyor."""
    profile = Profile(channels={"eyeBlinkLeft": ChannelProfile(input_max=0.917)})
    assert profile.apply({"eyeBlinkLeft": 0.917})["eyeBlinkLeft"] == pytest.approx(1.0, abs=1e-6)


def test_input_max_ara_degerleri_olcekliyor():
    profile = Profile(channels={"eyeBlinkLeft": ChannelProfile(input_max=0.8)})
    assert profile.apply({"eyeBlinkLeft": 0.4})["eyeBlinkLeft"] == pytest.approx(0.5)


def test_ogrenici_tepeyi_yakaliyor():
    learner = RangeLearner()
    for value in (0.2, 0.9, 0.5, 0.87):
        learner.feed({"eyeBlinkLeft": value})
    assert learner.maxima["eyeBlinkLeft"] == pytest.approx(0.9)
    assert learner.channel_profiles()["eyeBlinkLeft"].input_max == pytest.approx(0.9)


def test_ogrenici_zayif_kanallara_dokunmuyor():
    """Kanal hic tetiklenmediyse tepesi ~0.05 kalir; onu 1.0'a cekmek
    20x kazanc demek olur ve gurultuyu patlatir."""
    learner = RangeLearner(floor=0.35)
    for _ in range(10):
        learner.feed({"cheekPuff": 0.05})
    assert "cheekPuff" not in learner.channel_profiles()


def test_ogrenici_mevcut_ayari_koruyor():
    learner = RangeLearner()
    learner.feed({"jawOpen": 0.9})
    base = {"jawOpen": ChannelProfile(gain=2.0, deadzone=0.1)}
    result = learner.channel_profiles(base)["jawOpen"]
    assert result.gain == 2.0 and result.deadzone == 0.1
    assert result.input_max == pytest.approx(0.9)


# --------------------------------------------------------------------------
# buzusturme (opucuk) -- agzin ortasinda bosluk kalmasi
# --------------------------------------------------------------------------


def test_buzusturmede_dudaklar_birlesiyor():
    """Epic'in tablosunda mouthPucker funnel'i (acik O) suruyor ama hicbir
    sey dudaklari kapatmiyordu -> opucukte agzin ortasinda bosluk."""
    from core.realism import LIPS_TOGETHER_AXES

    layer = RealismLayer(saccades=0.0)
    extra = layer.extra_axes({"CTRL_L_mouth_purseU.y": 0.9})
    for key in LIPS_TOGETHER_AXES:
        assert extra[key] > 0.0, key


def test_cene_acikken_dudaklar_birlesmiyor():
    """Agiz acikken dudaklar fiziksel olarak birlesemez."""
    from core.realism import LIPS_TOGETHER_AXES

    layer = RealismLayer(saccades=0.0)
    kapali = layer.extra_axes({"CTRL_L_mouth_purseU.y": 0.9, "CTRL_C_jaw.y": 0.0})
    acik = layer.extra_axes({"CTRL_L_mouth_purseU.y": 0.9, "CTRL_C_jaw.y": 1.0})
    assert acik.get(LIPS_TOGETHER_AXES[0], 0.0) < kapali[LIPS_TOGETHER_AXES[0]]
    assert acik.get(LIPS_TOGETHER_AXES[0], 0.0) == pytest.approx(0.0)


def test_buzusturme_yoksa_dokunmuyor():
    from core.realism import LIPS_TOGETHER_AXES

    layer = RealismLayer(saccades=0.0)
    extra = layer.extra_axes({"CTRL_C_jaw.y": 0.5})
    assert LIPS_TOGETHER_AXES[0] not in extra


def test_mouth_close_daha_guclu_ise_eziyor():
    """mouthClose'dan gelen gercek olcum, turetilmis degerden buyukse
    korunmali -- turetme olcumu bastirmamali."""
    from core.realism import LIPS_TOGETHER_AXES

    layer = RealismLayer(saccades=0.0, pucker_close=0.5)
    extra = layer.extra_axes({"CTRL_L_mouth_purseU.y": 0.4, LIPS_TOGETHER_AXES[0]: 0.95})
    assert extra[LIPS_TOGETHER_AXES[0]] == pytest.approx(0.95)


def test_buzusturme_kapatilabiliyor():
    from core.realism import LIPS_TOGETHER_AXES

    layer = RealismLayer(saccades=0.0, pucker_close=0.0)
    extra = layer.extra_axes({"CTRL_L_mouth_purseU.y": 0.9})
    assert LIPS_TOGETHER_AXES[0] not in extra


# --------------------------------------------------------------------------
# aralik kaliciligi -- ROM cekimi Blender kapaninca kaybolmamali
# --------------------------------------------------------------------------


def test_ogrenilen_aralik_kaydedilip_yuklenebiliyor(tmp_path):
    learner = RangeLearner()
    for value in (0.4, 0.917, 0.6):
        learner.feed({"eyeBlinkLeft": value})
    learner.feed({"jawOpen": 0.973})

    path = tmp_path / "profil.json"
    learner.save(path)

    yeniden = Profile.load(path)
    assert yeniden.channels["eyeBlinkLeft"].input_max == pytest.approx(0.917)
    assert yeniden.channels["jawOpen"].input_max == pytest.approx(0.973)


def test_kaydetme_mevcut_profili_ezmiyor(tmp_path):
    """Kullanicinin elle ayarladigi kazanc/olu bolge korunmali."""
    path = tmp_path / "profil.json"
    Profile(channels={"jawOpen": ChannelProfile(gain=1.8, deadzone=0.12)}).save(path)

    learner = RangeLearner()
    learner.feed({"jawOpen": 0.9})
    learner.save(path)

    yeniden = Profile.load(path)
    assert yeniden.channels["jawOpen"].gain == pytest.approx(1.8)
    assert yeniden.channels["jawOpen"].deadzone == pytest.approx(0.12)
    assert yeniden.channels["jawOpen"].input_max == pytest.approx(0.9)


def test_kaydetme_dokunulmayan_kanallari_koruyor(tmp_path):
    path = tmp_path / "profil.json"
    Profile(channels={"mouthPucker": ChannelProfile(gain=0.6)}).save(path)

    learner = RangeLearner()
    learner.feed({"jawOpen": 0.9})
    learner.save(path)

    yeniden = Profile.load(path)
    assert "mouthPucker" in yeniden.channels
    assert yeniden.channels["mouthPucker"].gain == pytest.approx(0.6)


def test_yuklenen_profil_uygulanabiliyor(tmp_path):
    """Kaydet-yukle turu sonrasi normalize gercekten calismali."""
    learner = RangeLearner()
    learner.feed({"eyeBlinkLeft": 0.917})
    path = tmp_path / "profil.json"
    learner.save(path)

    profile = Profile.load(path)
    assert profile.apply({"eyeBlinkLeft": 0.917})["eyeBlinkLeft"] == pytest.approx(1.0, abs=1e-6)
