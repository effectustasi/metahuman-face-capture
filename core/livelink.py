"""Live Link Face (iPhone/ARKit) UDP paketlerini cozer.

MediaPipe'a gore ne kazandiriyor:

* TrueDepth sensoru -> RGB tahmininden belirgin daha kararli
* **tongueOut** var (MediaPipe'ta yok)
* **mouthClose** gercek olcum (bizde elle yazilmisti)
* ayrica goz yonu ve kafa rotasyonu ayri kanallar olarak geliyor

Kanal sirasi EZBERDEN YAZILMADI: gercek Live Link Face CSV ciktisinin
baslik satirindan alindi (`JawRight`in `JawLeft`ten once gelmesi gibi
sezgiye aykiri detaylar var).

Paket duzeni:

    uint8         surum
    int32 + bytes cihaz kimligi   (uzunluk onekli, big-endian)
    int32 + bytes konu adi
    int32         kare numarasi
    float         alt kare
    int32         fps
    int32         payda
    uint8         blendshape sayisi (61)
    float[61]     degerler          (big-endian)

Duzen Epic tarafindan resmi olarak belgelenmemis. Bu yuzden cozucu
kendini DOGRULUYOR: yapisal cozum tutmazsa paketin sonundan geriye
sayarak blendshape blogunu buluyor. Hangi yolun kullanildigini
`decode` sonucu bildiriyor, `scripts/llf_probe.py` de gosteriyor.
"""

from __future__ import annotations

import math
import struct

# Gercek Live Link Face CSV basligindan alindi. Sira onemli ve
# sezgiye aykiri: JawRight < JawLeft, MouthRight < MouthLeft.
CHANNELS = (
    "EyeBlinkLeft", "EyeLookDownLeft", "EyeLookInLeft", "EyeLookOutLeft", "EyeLookUpLeft",
    "EyeSquintLeft", "EyeWideLeft",
    "EyeBlinkRight", "EyeLookDownRight", "EyeLookInRight", "EyeLookOutRight", "EyeLookUpRight",
    "EyeSquintRight", "EyeWideRight",
    "JawForward", "JawRight", "JawLeft", "JawOpen",
    "MouthClose", "MouthFunnel", "MouthPucker", "MouthRight", "MouthLeft",
    "MouthSmileLeft", "MouthSmileRight", "MouthFrownLeft", "MouthFrownRight",
    "MouthDimpleLeft", "MouthDimpleRight", "MouthStretchLeft", "MouthStretchRight",
    "MouthRollLower", "MouthRollUpper", "MouthShrugLower", "MouthShrugUpper",
    "MouthPressLeft", "MouthPressRight", "MouthLowerDownLeft", "MouthLowerDownRight",
    "MouthUpperUpLeft", "MouthUpperUpRight",
    "BrowDownLeft", "BrowDownRight", "BrowInnerUp", "BrowOuterUpLeft", "BrowOuterUpRight",
    "CheekPuff", "CheekSquintLeft", "CheekSquintRight",
    "NoseSneerLeft", "NoseSneerRight",
    "TongueOut",
    "HeadYaw", "HeadPitch", "HeadRoll",
    "LeftEyeYaw", "LeftEyePitch", "LeftEyeRoll",
    "RightEyeYaw", "RightEyePitch", "RightEyeRoll",
)

BLENDSHAPE_COUNT = 52  # ilk 52 ifade kanali
CHANNEL_COUNT = len(CHANNELS)  # 61 (52 + kafa 3 + goz 6)

HEAD_CHANNELS = ("HeadYaw", "HeadPitch", "HeadRoll")


class DecodeError(ValueError):
    pass


def to_mediapipe_name(name: str) -> str:
    """'EyeBlinkLeft' -> 'eyeBlinkLeft'. Mapping tablosu MediaPipe
    isimleriyle anahtarli; iki kaynak da ayni sozluge donusuyor."""
    return name[0].lower() + name[1:] if name else name


def _read_prefixed(data: bytes, offset: int) -> tuple[bytes, int]:
    if offset + 4 > len(data):
        raise DecodeError("uzunluk oneki icin yer yok")
    (size,) = struct.unpack_from("!i", data, offset)
    offset += 4
    if size < 0 or offset + size > len(data):
        raise DecodeError(f"gecersiz uzunluk oneki: {size}")
    return data[offset : offset + size], offset + size


def _values_are_sane(values) -> bool:
    """Blendshape'ler [0,1], kafa/goz acilari radyan. Hepsi sonlu ve
    makul araliktaysa dogru yeri bulmusuz demektir."""
    for value in values:
        if not math.isfinite(value) or abs(value) > 100.0:
            return False
    expressions = values[:BLENDSHAPE_COUNT]
    return all(-0.01 <= v <= 1.01 for v in expressions)


def _decode_structured(data: bytes) -> tuple[list[float], dict]:
    offset = 1  # surum baytini atla
    device_id, offset = _read_prefixed(data, offset)
    subject, offset = _read_prefixed(data, offset)

    if offset + 17 > len(data):
        raise DecodeError("timecode + sayac icin yer yok")
    frame_number, sub_frame, fps, denominator = struct.unpack_from("!ifii", data, offset)
    offset += 16
    (count,) = struct.unpack_from("!B", data, offset)
    offset += 1

    if count != CHANNEL_COUNT or offset + count * 4 != len(data):
        raise DecodeError(f"beklenmeyen kanal sayisi/uzunluk: {count}")

    values = list(struct.unpack_from(f"!{count}f", data, offset))
    if not _values_are_sane(values):
        raise DecodeError("degerler makul aralikta degil")

    return values, {
        "method": "structured",
        "version": data[0],
        "device": device_id.decode("utf-8", "replace"),
        "subject": subject.decode("utf-8", "replace"),
        "frame": frame_number,
        "subFrame": sub_frame,
        "fps": fps,
        "denominator": denominator,
    }


def _decode_by_scan(data: bytes) -> tuple[list[float], dict]:
    """Basliga guvenmeden sondan geri say.

    Paketin son `1 + 61*4` bayti sayac + degerler olmali. Duzen Epic
    tarafindan belgelenmedigi icin basligin degismesine karsi bu yol
    duruyor -- gercek telefonla dogrulanmadan once tek guvencemiz bu.
    """
    block = 1 + CHANNEL_COUNT * 4
    if len(data) < block:
        raise DecodeError("paket cok kisa")

    offset = len(data) - block
    (count,) = struct.unpack_from("!B", data, offset)
    if count != CHANNEL_COUNT:
        raise DecodeError(f"sondan sayacta {count} bulundu, {CHANNEL_COUNT} bekleniyordu")

    values = list(struct.unpack_from(f"!{CHANNEL_COUNT}f", data, offset + 1))
    if not _values_are_sane(values):
        raise DecodeError("sondan okunan degerler makul degil")

    return values, {"method": "scan", "version": data[0] if data else None}


def decode(data: bytes) -> tuple[dict[str, float], dict]:
    """Ham UDP paketi -> ({kanal adi: deger}, meta).

    Kanal adlari Live Link'in PascalCase'i olarak kalir; MediaPipe
    ismine cevirmek `to_frame`in isi.
    """
    errors = []
    for parser in (_decode_structured, _decode_by_scan):
        try:
            values, meta = parser(data)
        except (DecodeError, struct.error) as error:
            errors.append(f"{parser.__name__}: {error}")
            continue
        return dict(zip(CHANNELS, values)), meta

    raise DecodeError("cozulemedi -> " + " | ".join(errors))


def head_quaternion(channels: dict[str, float]) -> list[float] | None:
    """HeadYaw/Pitch/Roll (radyan) -> quaternion (x, y, z, w).

    Neden quaternion'a ceviriyoruz: core.head.HeadTracker MediaPipe'in
    quaternion'unu alip yaw/pitch/roll'a ayristiriyor. Ayni yola
    sokarsak kullanicinin kalibre ettigi eksen eslemesi, notr ogrenme
    ve kazanclar iki kaynak icin de AYNEN calisir. Tek kod yolu.
    """
    if not all(name in channels for name in HEAD_CHANNELS):
        return None

    yaw = channels["HeadYaw"]
    pitch = channels["HeadPitch"]
    roll = channels["HeadRoll"]

    # core.head.decompose ile ayni sozlesme: pitch=X, yaw=Y, roll=Z
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)

    return [
        sp * cy * cr + cp * sy * sr,
        cp * sy * cr - sp * cy * sr,
        cp * cy * sr - sp * sy * cr,
        cp * cy * cr + sp * sy * sr,
    ]


def to_frame_payload(data: bytes, timestamp: float, frame_number: int) -> dict:
    """Ham paket -> core.take semasindaki sozluk.

    Boylece Blender tarafi iPhone ile webcam arasinda fark gormuyor.
    """
    channels, meta = decode(data)

    blendshapes = {
        to_mediapipe_name(name): value
        for name, value in channels.items()
        if name in CHANNELS[:BLENDSHAPE_COUNT]
    }

    payload = {
        "t": round(timestamp, 6),
        "frame": meta.get("frame") or frame_number,
        "bs": blendshapes,
    }

    rotation = head_quaternion(channels)
    if rotation is not None:
        payload["head"] = {"rot": rotation, "trans": [0.0, 0.0, 0.0]}

    return payload
