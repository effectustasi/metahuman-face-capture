"""Live Link Face cozucusu testleri.

Gercek telefon olmadan dogrulanabilen kisim: paket duzeni, kanal sirasi,
kafa acisi donusumu ve bozuk pakete dayaniklilik. Duzenin gercekten
boyle oldugu ancak telefonla `scripts/llf_probe.py` calistirilinca
kesinlesir -- o yuzden cozucude sondan tarayan yedek yol var.
"""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.head import decompose  # noqa: E402
from core.livelink import (  # noqa: E402
    CHANNEL_COUNT,
    CHANNELS,
    DecodeError,
    decode,
    head_quaternion,
    to_frame_payload,
    to_mediapipe_name,
)
from core.mapping import Mapping  # noqa: E402


def build_packet(values, device="iPhone15", subject="Ahmet", frame=42, corrupt_header=False):
    """Belgelenmis duzene gore sentetik paket uret."""
    payload = bytearray()
    payload += struct.pack("!B", 6)
    for text in (device, subject):
        raw = text.encode("utf-8")
        payload += struct.pack("!i", len(raw)) + raw
    payload += struct.pack("!ifii", frame, 0.0, 60, 1)
    payload += struct.pack("!B", len(values))
    payload += struct.pack(f"!{len(values)}f", *values)

    if corrupt_header:
        # basligi boz ama blendshape blogunu birak -> yedek yol devreye girmeli
        payload[1:5] = struct.pack("!i", 999999)

    return bytes(payload)


@pytest.fixture
def values():
    v = [0.0] * CHANNEL_COUNT
    v[CHANNELS.index("JawOpen")] = 0.7
    v[CHANNELS.index("MouthClose")] = 0.4
    v[CHANNELS.index("TongueOut")] = 0.9
    v[CHANNELS.index("EyeBlinkLeft")] = 0.2
    v[CHANNELS.index("HeadYaw")] = math.radians(20)
    v[CHANNELS.index("HeadPitch")] = math.radians(-10)
    v[CHANNELS.index("HeadRoll")] = math.radians(5)
    return v


# --------------------------------------------------------------------------
# kanal sirasi -- gercek CSV basligindan alindi, degismemeli
# --------------------------------------------------------------------------


def test_kanal_sayisi():
    assert CHANNEL_COUNT == 61, "52 ifade + 3 kafa + 6 goz"


def test_jaw_ve_mouth_sirasi_sezgiye_aykiri():
    """JawRight, JawLeft'ten ONCE geliyor. Ezberden yazilirsa ters olur
    ve cene sag/sol takla atar."""
    assert CHANNELS.index("JawRight") < CHANNELS.index("JawLeft")
    assert CHANNELS.index("MouthRight") < CHANNELS.index("MouthLeft")


def test_tongue_out_var():
    assert "TongueOut" in CHANNELS, "iPhone'un MediaPipe'a gore artisi"


# --------------------------------------------------------------------------
# cozme
# --------------------------------------------------------------------------


def test_yapisal_cozum(values):
    channels, meta = decode(build_packet(values))
    assert meta["method"] == "structured"
    assert meta["device"] == "iPhone15"
    assert meta["frame"] == 42
    assert channels["JawOpen"] == pytest.approx(0.7)
    assert channels["TongueOut"] == pytest.approx(0.9)


def test_baslik_bozuksa_sondan_tarayarak_cozuyor(values):
    channels, meta = decode(build_packet(values, corrupt_header=True))
    assert meta["method"] == "scan"
    assert channels["JawOpen"] == pytest.approx(0.7)
    assert channels["MouthClose"] == pytest.approx(0.4)


def test_bozuk_paket_hata_veriyor():
    with pytest.raises(DecodeError):
        decode(b"\x06kisa")
    with pytest.raises(DecodeError):
        decode(b"")


def test_makul_olmayan_degerler_reddediliyor():
    """Blendshape [0,1] disindaysa yanlis yeri okumusuzdur; sessizce
    kabul etmek karakteri patlatir."""
    bad = [5.0] * CHANNEL_COUNT
    with pytest.raises(DecodeError):
        decode(build_packet(bad))


# --------------------------------------------------------------------------
# semaya donusum
# --------------------------------------------------------------------------


def test_isim_donusumu():
    assert to_mediapipe_name("EyeBlinkLeft") == "eyeBlinkLeft"
    assert to_mediapipe_name("TongueOut") == "tongueOut"


def test_frame_payload_semasi(values):
    payload = to_frame_payload(build_packet(values), timestamp=1.5, frame_number=1)
    assert payload["t"] == pytest.approx(1.5)
    assert payload["frame"] == 42
    assert "head" in payload
    assert len(payload["bs"]) == 52, "sadece ifade kanallari, kafa/goz ayri"
    assert payload["bs"]["jawOpen"] == pytest.approx(0.7)
    assert "headYaw" not in payload["bs"], "kafa kanallari blendshape'e sizmamali"


def test_cikti_mapping_tablosuyla_uyumlu(values):
    """iPhone ciktisi tabloya dogrudan girmeli -- cevirme katmani yok."""
    payload = to_frame_payload(build_packet(values), timestamp=0.0, frame_number=1)
    mapping = Mapping.load()
    assert mapping.unknown_shapes(payload["bs"]) == []
    sonuc = mapping.apply(payload["bs"])
    assert sonuc, "hicbir eksen surulmedi"
    assert "CTRL_C_jaw.y" in sonuc


def test_iphone_mouth_close_suruyor(values):
    """MediaPipe'ta olmayan, elle yazdigimiz kanal iPhone'da gercek
    olcum olarak geliyor."""
    payload = to_frame_payload(build_packet(values), timestamp=0.0, frame_number=1)
    sonuc = Mapping.load().apply(payload["bs"])
    assert any("lipsTogether" in eksen for eksen in sonuc), "mouthClose surulmedi"


# --------------------------------------------------------------------------
# kafa acisi
# --------------------------------------------------------------------------


def test_kafa_acilari_gidip_geliyor(values):
    """LLF yaw/pitch/roll -> quaternion -> core.head.decompose ayni
    acilari geri vermeli. Ayni kod yolunu kullandigimiz icin kullanicinin
    kalibrasyonu iki kaynak icin de aynen calisiyor."""
    channels, _ = decode(build_packet(values))
    q = head_quaternion(channels)
    pitch, yaw, roll = decompose(q)
    assert yaw == pytest.approx(math.radians(20), abs=1e-4)
    assert pitch == pytest.approx(math.radians(-10), abs=1e-4)
    assert roll == pytest.approx(math.radians(5), abs=1e-4)


def test_kafa_kanali_yoksa_none():
    assert head_quaternion({"JawOpen": 0.5}) is None


# --------------------------------------------------------------------------
# GERCEK iPhone verisi
# --------------------------------------------------------------------------

REAL_PACKET = ROOT / "tests" / "fixtures" / "livelink_frame.bin"


@pytest.mark.skipif(not REAL_PACKET.exists(), reason="gercek paket fixture'i yok")
def test_gercek_iphone_paketi_cozuluyor():
    """Gercek bir iPhone'dan yakalanmis paket (Animoji reposundaki kayittan;
    cihaz kimligi ve telefon adi ayni uzunlukta yer tutucuyla degistirildi).

    Bu test cozucunun belgelenmemis paket duzenini dogru kurdugunun
    kanitidir -- telefon olmadan dogrulanabilen tek nokta buydu.
    """
    channels, meta = decode(REAL_PACKET.read_bytes())

    assert meta["method"] == "structured", "yapisal cozum tutmali, yedek yola dusmemeli"
    assert meta["fps"] == 60
    assert len(channels) == CHANNEL_COUNT

    # kayitta denek dilini cikarmis: MediaPipe'in asla uretemeyecegi kanal
    assert channels["TongueOut"] == pytest.approx(1.0, abs=1e-3)
    assert 0.0 <= channels["JawOpen"] <= 1.0
    assert channels["MouthClose"] > 0.0


@pytest.mark.skipif(not REAL_PACKET.exists(), reason="gercek paket fixture'i yok")
def test_gercek_paket_zincirin_sonuna_kadar_gidiyor():
    payload = to_frame_payload(REAL_PACKET.read_bytes(), timestamp=0.0, frame_number=1)
    mapping = Mapping.load()

    assert mapping.unknown_shapes(payload["bs"]) == []
    values = mapping.apply(payload["bs"])
    assert values

    # dil disari -> dil kontrolleri surulmeli
    assert any("tongue" in eksen.lower() for eksen in values), "tongueOut rig'e ulasmadi"
