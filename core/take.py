"""Take (cekim) dosya formati: satir basina bir JSON nesnesi (.jsonl).

Ayni sema hem offline dosyada hem canli UDP paketinde kullanilir; Blender
tarafi ikisini de ayni kodla okur.

    {"t": 0.0333, "frame": 1,
     "bs": {"jawOpen": 0.42, "eyeBlinkLeft": 0.03, ...},
     "head": {"rot": [x, y, z, w], "trans": [x, y, z]}}

  t     : saniye, cekim basindan itibaren
  frame : 1'den baslayan kare numarasi
  bs    : ARKit blendshape skorlari, MediaPipe kategori isimleriyle
          ('_neutral' yazilmaz). Sifir olanlar atlanabilir.
  head  : kafa pozu. rot = quaternion (x, y, z, w). Blendshape'lerden
          AYRI kanal: head/neck kemigine uygulanir, face board'a degil.
          Yoksa None.
  lm    : ham MediaPipe landmarklari, [[x, y, z], ...] 478 adet. Normalize
          goruntu kordinati (x, y in [0,1]), z gorece derinlik -- METRIK
          DEGIL, o yuzden 3B karsilastirmada olcek serbest birakilmali.
          Sadece cozucu (core/solver.py) icin gerekiyor, ARKit yolu
          kullanmiyor; bu yuzden istege bagli (`--landmarks`) ve varsayilan
          olarak yazilmiyor -- kare basina ~10 KB, cekimi 30 katina cikarir.
  res   : [genislik, yukseklik] piksel. `lm` ile birlikte yazilir. MediaPipe
          x'i GENISLIGE, y'yi YUKSEKLIGE bolerek normalize ediyor; kare
          olmayan goruntude ayni normalize fark farkli piksel demek. Oran
          bilinmeden landmark'lar carpik okunur.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class Frame:
    t: float
    frame: int
    bs: dict[str, float] = field(default_factory=dict)
    head: dict | None = None
    lm: list[list[float]] | None = None
    res: list[int] | None = None

    @classmethod
    def from_json(cls, payload: dict) -> "Frame":
        return cls(
            t=float(payload["t"]),
            frame=int(payload["frame"]),
            bs={k: float(v) for k, v in (payload.get("bs") or {}).items()},
            head=payload.get("head"),
            lm=payload.get("lm"),
            res=payload.get("res"),
        )

    def to_json(self) -> dict:
        out: dict = {"t": round(self.t, 6), "frame": self.frame, "bs": self.bs}
        if self.head is not None:
            out["head"] = self.head
        if self.lm is not None:
            out["lm"] = self.lm
            out["res"] = self.res
        return out


def read_take(path: Path | str) -> Iterator[Frame]:
    """Bos satirlari ve bozuk satirlari atlamaz -- bozuk satir hatadir,
    sessizce yutulursa hangi karenin kayip oldugu anlasilmaz."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Frame.from_json(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number} bozuk take satiri: {error}") from error


def write_take(path: Path | str, frames) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for frame in frames:
            handle.write(json.dumps(frame.to_json(), separators=(",", ":")) + "\n")
            count += 1
    return count


def plan_scene_frames(entries, fps: float, start_frame: int = 1) -> dict:
    """Cekim karelerini ZAMAN DAMGASINA gore sahne karelerine oturtur.

    `entries`: (t, blendshapes, head) uclulerinin sirali listesi. Profil ve
    filtre BURAYA GELMEDEN uygulanmis olmali -- one-euro cekim sirasina
    bagli, birlestirilmis kareler uzerinde calismaz.

    Neden kare kareye 1:1 yazilmiyor: webcam cekimi 30 fps, Blender'in
    varsayilan sahnesi 24 fps. Olculdu -- 37 saniyelik cekim 1:1 yazilinca
    46 saniye suruyor, %25 yavas. Kare araligi da sabit degil (12-50 ms),
    MediaPipe yetisemedigi anlarda geciktiriyor.

    Ayni sahne karesine birden fazla cekim karesi dustugunde kanallarin EN
    YUKSEGI aliniyor. 30->24'te her bes kareden biri birlesiyor; sonuncuyu
    almak goz kirpma tepesini kaybettiriyor -- kirpma 89 ms surerken 42 ms'lik
    bir kare araligi tepenin tam ustune denk gelebiliyor.

    Zaman damgasi kullanilamaz durumdaysa (hepsi ayni, ya da geri gidiyor)
    sirali yazima duser; sessizce yanlis hizda animasyon uretmektense
    eski davranisi korumak yeglenir.
    """
    entries = list(entries)
    if not entries:
        return {}

    times = [t for t, _, _ in entries]
    usable = times[-1] > times[0] and all(b >= a for a, b in zip(times, times[1:]))
    origin = times[0]

    plan: dict[int, list] = {}
    for offset, (t, blendshapes, head) in enumerate(entries):
        frame = start_frame + (round((t - origin) * fps) if usable else offset)
        slot = plan.get(frame)
        if slot is None:
            plan[frame] = [dict(blendshapes), head]
            continue
        for channel, value in blendshapes.items():
            if value > slot[0].get(channel, 0.0):
                slot[0][channel] = value
        if head:
            slot[1] = head
    return {frame: (values, head) for frame, (values, head) in plan.items()}
