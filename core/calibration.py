"""Notr yuz kalibrasyonu ve kanal basina kazanc/olu bolge.

Neden gerekli: MediaPipe hicbir zaman tam sifir dondurmez. Kisinin dinlenme
yuzunde jawOpen 0.05, mouthPucker 0.08 gibi degerler oturur; bunlar dogrudan
rig'e gidince karakter surekli hafif agzi acik durur. Notr offset'i cikarip
kalan araligi yeniden olceklemek bunu duzeltir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


# Kanal gruplari. Kanal kanal ayar yapmak 51 kaydirici demek; pratikte
# insanlar "agiz fazla oynuyor" ya da "kaslar olu" diye dusunuyor, tek tek
# kanal diye degil. Gruplar bu dili karsiliyor.
CHANNEL_GROUPS: dict[str, tuple[str, ...]] = {
    "jaw": ("jawOpen", "jawForward", "jawLeft", "jawRight"),
    "lipClose": ("mouthClose",),
    "lips": (),  # asagida doldurulur: kalan tum mouth* kanallari
    "eyes": (),
    "brows": (),
    "cheeksNose": (),
}


def group_of(channel: str) -> str:
    """Kanalin hangi gruba dustugu. Isim onekine bakar; MediaPipe isimleri
    zaten bolgeye gore onekli oldugu icin el tablosu gerekmiyor."""
    if channel in CHANNEL_GROUPS["jaw"]:
        return "jaw"
    if channel in CHANNEL_GROUPS["lipClose"]:
        return "lipClose"
    if channel.startswith("mouth"):
        return "lips"
    if channel.startswith("eye"):
        return "eyes"
    if channel.startswith("brow"):
        return "brows"
    if channel.startswith(("cheek", "nose")):
        return "cheeksNose"
    return "other"


@dataclass
class ChannelProfile:
    gain: float = 1.0
    deadzone: float = 0.0
    clamp_min: float = 0.0
    clamp_max: float = 1.0
    # Gozlenen giris araligi. Kanal 1.0'a hic ulasmiyorsa rig tam acilmiyor.
    # Gercek iPhone kaydinda olculdu: eyeBlink 0.917'de doyuyor, mouthSmile
    # 0.865, browInnerUp 0.858 -- yani goz hicbir zaman tam kapanmiyor.
    # "Olu bakis" hissinin somut kaynagi bu.
    input_max: float = 1.0


@dataclass
class Profile:
    """profile.json'un calisir hali."""

    neutral: dict[str, float] = field(default_factory=dict)
    channels: dict[str, ChannelProfile] = field(default_factory=dict)
    groups: dict[str, ChannelProfile] = field(default_factory=dict)
    default: ChannelProfile = field(default_factory=ChannelProfile)
    mirror: bool = False

    @classmethod
    def load(cls, path: Path | str) -> "Profile":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            neutral=payload.get("neutral", {}),
            channels={k: ChannelProfile(**v) for k, v in payload.get("channels", {}).items()},
            groups={k: ChannelProfile(**v) for k, v in payload.get("groups", {}).items()},
            default=ChannelProfile(**payload.get("default", {})),
            mirror=payload.get("mirror", False),
        )

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "neutral": {k: round(v, 5) for k, v in self.neutral.items()},
                    "channels": {k: vars(v) for k, v in self.channels.items()},
                    "groups": {k: vars(v) for k, v in self.groups.items()},
                    "default": vars(self.default),
                    "mirror": self.mirror,
                },
                indent=1,
            ),
            encoding="utf-8",
        )

    def profile_for(self, channel: str) -> ChannelProfile:
        """Oncelik: kanala ozel > grup > genel varsayilan."""
        specific = self.channels.get(channel)
        if specific is not None:
            return specific
        grouped = self.groups.get(group_of(channel))
        if grouped is not None:
            return grouped
        return self.default

    def apply(self, blendshapes: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, value in blendshapes.items():
            profile = self.profile_for(name)

            # notr offset'i cikar, kalan araligi [0,1]'e geri yay
            offset = self.neutral.get(name, 0.0)
            if offset > 0.0 and offset < 1.0:
                value = (value - offset) / (1.0 - offset)

            # gozlenen tepeyi 1.0'a normalize et -- yoksa rig hic tam
            # acilmiyor (bkz. ChannelProfile.input_max)
            if 0.0 < profile.input_max < 1.0:
                value = value / profile.input_max

            if value < profile.deadzone:
                value = 0.0

            value *= profile.gain
            out[name] = max(profile.clamp_min, min(profile.clamp_max, value))

        return mirror_blendshapes(out) if self.mirror else out


class RangeLearner:
    """Kanal basina gozlenen tepeyi ogrenir (ROM cekimi icin).

    Kullanim: kullanici tum ifadeleri sonuna kadar yapar, ogrenilen
    tepeler profile yazilir. Sonra her kanal kendi gercek araligina gore
    normalize edilir ve rig tam acilir.
    """

    def __init__(self, floor: float = 0.35):
        # Cok dusuk tepeleri normalize etmek gurultuyu patlatir: kanal
        # hic tetiklenmediyse tepesi ~0.05 kalir ve 20x kazanc uygulanir.
        self.floor = floor
        self.maxima: dict[str, float] = {}
        self.samples = 0

    def feed(self, blendshapes: dict[str, float]) -> None:
        self.samples += 1
        for name, value in blendshapes.items():
            if value > self.maxima.get(name, 0.0):
                self.maxima[name] = value

    def channel_profiles(self, base: dict[str, ChannelProfile] | None = None) -> dict[str, ChannelProfile]:
        result = dict(base or {})
        for name, peak in self.maxima.items():
            if peak < self.floor:
                continue  # yeterince tetiklenmemis, dokunma
            profile = result.get(name)
            if profile is None:
                profile = ChannelProfile()
            result[name] = ChannelProfile(
                gain=profile.gain,
                deadzone=profile.deadzone,
                clamp_min=profile.clamp_min,
                clamp_max=profile.clamp_max,
                input_max=round(peak, 4),
            )
        return result

    def save(self, path: Path | str) -> int:
        """Ogrenilen araliklari profile yaz. Kac kanal yazildigini doner.

        ROM cekimi emek isteyen bir is; Blender kapaninca kaybolmamali.
        """
        profile = Profile()
        path = Path(path)
        if path.exists():
            profile = Profile.load(path)  # mevcut ayarlari koru
        profile.channels = self.channel_profiles(profile.channels)
        profile.save(path)
        return len(profile.channels)

    def summary(self) -> str:
        learned = sum(1 for v in self.maxima.values() if v >= self.floor)
        return f"{self.samples} kare, {learned}/{len(self.maxima)} kanal ogrenildi"


def build_neutral(frames, sample_count: int = 30) -> dict[str, float]:
    """Ilk N karenin ortalamasini notr kabul et. Cekimin basinda deneğin
    ifadesiz durmasi gerekiyor -- bu sartlanmis bir varsayim, cekim
    talimatinda soylenmeli."""
    totals: dict[str, float] = {}
    used = 0
    for frame in frames:
        if used >= sample_count:
            break
        for name, value in frame.bs.items():
            totals[name] = totals.get(name, 0.0) + value
        used += 1
    if not used:
        return {}
    return {name: total / used for name, total in totals.items()}


def mirror_blendshapes(blendshapes: dict[str, float]) -> dict[str, float]:
    """Sol/sag takas. Kamera goruntusunde kullanicinin solu ekranda sagda
    gorunur; karakterin 'sol'u ile eslesmesi isteniyorsa bu bayrak acilir."""
    out = {}
    for name, value in blendshapes.items():
        if name.endswith("Left"):
            out[name[: -len("Left")] + "Right"] = value
        elif name.endswith("Right"):
            out[name[: -len("Right")] + "Left"] = value
        else:
            out[name] = value
    return out
