"""Kafa pozu: MediaPipe quaternion'undan Blender kemik rotasyonuna.

Neden ayri bir katman gerekiyor:

1. **Mutlak degil bagil.** MediaPipe kameraya gore mutlak yonelim veriyor.
   Bunu dogrudan kemige yazarsan karakterin kafasi kameranin onunde nasil
   duruyorsan oyle sabitlenir. Dogrusu: cekimin basindaki notr yonelimi
   referans alip ONA GORE farki uygulamak.

2. **Eksenler rig'e gore degisiyor.** MediaPipe'in yuz uzayi ile kemigin
   yerel uzayi ayni degil. Quaternion bileseni permutasyonu olarak ayara
   acmak ise ise yaramiyor -- "hangi permutasyon dogru" sorusunun goze
   bakarak cevabi yok.

   Bu yuzden once ANLAMLI acilara ayristiriyoruz (yaw = saga sola cevirme,
   pitch = asagi yukari, roll = yana yatirma), sonra her birini kullanicinin
   secitigi kemik eksenine yolluyoruz. Boylece belirti dogrudan ayara
   ceviriliyor: "saga cevirince asagi bakiyor" -> yaw yanlis eksende.

Stdlib disi bagimlilik yok; Blender'dan ve testlerden ayni sekilde import
edilir.
"""

from __future__ import annotations

import math

AXES = ("X", "Y", "Z")
AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}

# Gercek MetaHuman rig'inde OLCULDU, teorik degil. Iki gozlemle kuruldu:
#
#   1. yaw -> Z gonderildiginde karakter asagi/yukari bakti   =>  Z = PITCH
#   2. yaw -> Y, roll -> X iken ikisi birbirinin yerine calisti
#                                                            =>  Y = ROLL, X = YAW
#
# Yani kafa kemiginin yerel eksenleri:  X = yaw, Y = roll, Z = pitch.
#
# Bu Blender'in "kemik Y ekseni boyunca uzanir" kuralindan cikan sezgiye
# uymuyor (Y twist olsaydi yaw beklerdik); MetaHuman kemikleri UE'den
# geldigi icin yonelim farkli. Tam da bu yuzden kodda sabit degil, panelden
# ayarlanabilir -- baska bir rig'de baska cikabilir.
DEFAULT_MAPPING = {
    "yaw": ("X", 1.0),
    "pitch": ("Z", 1.0),
    "roll": ("Y", 1.0),
}


def normalize(q: list[float]) -> list[float]:
    """(x, y, z, w) birim uzunluga getir."""
    length = sum(component * component for component in q) ** 0.5
    if length < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [component / length for component in q]


def conjugate(q: list[float]) -> list[float]:
    x, y, z, w = q
    return [-x, -y, -z, w]


def multiply(a: list[float], b: list[float]) -> list[float]:
    """Hamilton carpimi, (x, y, z, w) sirasinda."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def relative_to(current: list[float], neutral: list[float]) -> list[float]:
    """Notr yonelime gore fark. neutral^-1 * current."""
    return normalize(multiply(conjugate(normalize(neutral)), normalize(current)))


def decompose(q: list[float]) -> tuple[float, float, float]:
    """Quaternion -> (pitch, yaw, roll) radyan, MediaPipe uzayinda.

    pitch = X ekseni etrafinda (basi one arkaya egme)
    yaw   = Y ekseni etrafinda (saga sola cevirme)
    roll  = Z ekseni etrafinda (omuza dogru yatirma)
    """
    x, y, z, w = normalize(q)

    sin_pitch = 2.0 * (w * x - y * z)
    sin_pitch = max(-1.0, min(1.0, sin_pitch))  # asin alan disina cikmasin
    pitch = math.asin(sin_pitch)

    yaw = math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (x * x + y * y))
    roll = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (x * x + z * z))

    return pitch, yaw, roll


def quaternion_from_axis(axis_index: int, angle: float) -> list[float]:
    half = angle * 0.5
    vector = [0.0, 0.0, 0.0]
    vector[axis_index] = math.sin(half)
    return vector + [math.cos(half)]


def compose(
    pitch: float,
    yaw: float,
    roll: float,
    mapping: dict[str, tuple[str, float]] | None = None,
) -> list[float]:
    """Acilari kemik uzayinda quaternion'a cevir.

    mapping: {"yaw": ("Z", -1.0), ...} -- her aci hangi kemik eksenine,
    hangi isaretle gidecek. 'axis' None/bos ise o aci uygulanmaz.
    """
    mapping = mapping or DEFAULT_MAPPING
    angles = {"pitch": pitch, "yaw": yaw, "roll": roll}

    result = [0.0, 0.0, 0.0, 1.0]
    # sabit sira: once yaw, sonra pitch, sonra roll. Kucuk acilarda sira
    # farki gorunmez, buyuk acilarda tutarli olmasi yeterli.
    for name in ("yaw", "pitch", "roll"):
        axis, sign = mapping.get(name, (None, 1.0))
        if not axis or axis not in AXIS_INDEX:
            continue
        result = multiply(result, quaternion_from_axis(AXIS_INDEX[axis], angles[name] * sign))

    return normalize(result)


def scale(q: list[float], influence: float) -> list[float]:
    """Birim quaternion ile aralarinda dogrusal karistir (nlerp).

    influence=0 -> hic donme, 1 -> tam donme. Ara degerler kafa hareketini
    yumusatmak icin; slerp'e gore ucuz ve bu aci araliginda farki gorunmez.
    """
    if influence >= 1.0:
        return normalize(q)
    if influence <= 0.0:
        return [0.0, 0.0, 0.0, 1.0]

    x, y, z, w = normalize(q)
    if w < 0.0:  # en kisa yol
        x, y, z, w = -x, -y, -z, -w
    return normalize(
        [x * influence, y * influence, z * influence, w * influence + (1.0 - influence)]
    )


def to_blender_quaternion(q: list[float]) -> tuple[float, float, float, float]:
    """(x, y, z, w) -> (w, x, y, z). Blender'in quaternion sirasi farkli."""
    x, y, z, w = q
    return (w, x, y, z)


class HeadTracker:
    """Notr yonelimi ogrenip bagil donmeyi kemik uzayinda uretir."""

    def __init__(
        self,
        mapping: dict[str, tuple[str, float]] | None = None,
        influence: float = 1.0,
        sample_count: int = 30,
    ):
        self.mapping = mapping or dict(DEFAULT_MAPPING)
        self.influence = influence
        self.sample_count = sample_count
        self._neutral: list[float] | None = None
        self._samples = 0
        self.last_angles: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def reset(self) -> None:
        self._neutral = None
        self._samples = 0

    def feed(self, rotation: list[float] | None) -> tuple[float, float, float, float] | None:
        """Ham MediaPipe quaternion'u ver, Blender quaternion'u al.

        Ilk `sample_count` kare notr referansi kurmak icin kullanilir; o
        sirada da cikti uretilir, boylece basta donma olmaz.
        """
        if not rotation or len(rotation) != 4:
            return None

        current = normalize([float(v) for v in rotation])

        if self._neutral is None:
            self._neutral = current
            self._samples = 1
        elif self._samples < self.sample_count:
            weight = 1.0 / (self._samples + 1)
            self._neutral = normalize(
                [n * (1.0 - weight) + c * weight for n, c in zip(self._neutral, current)]
            )
            self._samples += 1

        pitch, yaw, roll = decompose(relative_to(current, self._neutral))
        self.last_angles = (pitch, yaw, roll)
        return to_blender_quaternion(scale(compose(pitch, yaw, roll, self.mapping), self.influence))
