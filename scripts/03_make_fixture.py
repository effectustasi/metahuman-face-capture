"""tests/fixtures/take_5s.jsonl uretir: 5 saniye, 30 fps, sentetik hareket.

Gercek kamera olmadan bake yolunu bastan sona kosturabilmek icin. Icerik
kasitli olarak "zor" secildi: notr offset var (jawOpen hicbir zaman tam
sifira inmiyor), blink kisa darbeler halinde, ve blink/wide ayni eksende
cakisiyor.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.take import Frame, write_take  # noqa: E402

FPS = 30
DURATION = 5.0


def generate():
    for index in range(int(FPS * DURATION)):
        t = index / FPS
        phase = t * 2.0 * math.pi / 2.5  # 2.5 saniyelik dongu

        jaw = max(0.0, math.sin(phase)) * 0.8
        # notr taban: gercek MediaPipe ciktisi da tam sifira inmez
        jaw = 0.06 + jaw * 0.94

        # 1.2 sn'de bir ~4 karelik kirpma darbesi
        blink = 1.0 if (index % 36) < 4 else 0.0

        smile = max(0.0, math.sin(phase * 0.5)) * 0.6
        brow = max(0.0, math.sin(phase * 0.33 + 1.0)) * 0.5

        # goz acma, kirpmanin olmadigi anlarda -- ayni GUI eksenini
        # ters yonde suruyor, toplama mantigini sinar
        wide = 0.4 if (index % 90) in range(50, 60) else 0.0

        yield Frame(
            t=round(t, 6),
            frame=index + 1,
            bs={
                "jawOpen": round(jaw, 5),
                "eyeBlinkLeft": blink,
                "eyeBlinkRight": blink,
                "eyeWideLeft": wide,
                "eyeWideRight": wide,
                "mouthSmileLeft": round(smile, 5),
                "mouthSmileRight": round(smile, 5),
                "browOuterUpLeft": round(brow, 5),
                "browOuterUpRight": round(brow, 5),
            },
            head={"rot": [0.0, 0.0, 0.0, 1.0], "trans": [0.0, 0.0, 0.0]},
        )


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "take_5s.jsonl"
    count = write_take(out, generate())
    print(f"{count} kare -> {out}")
