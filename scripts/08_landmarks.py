"""Bir goruntudeki 478 yuz landmark'ini cikarir (detector venv'inde kosar).

Karsilik zincirinin 2. adimi:
  1. scripts/07_render_head.py       -> render + kamera (Blender)
  2. **bu script**                   -> landmark'lar (detector venv)
  3. scripts/09_build_correspondence.py -> isin at, vertex bul (Blender)

    detector/.venv/Scripts/python scripts/08_landmarks.py <goruntu.png>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "detector"))

from detect import build_landmarker, import_mediapipe  # noqa: E402

DEFAULT_MODEL = ROOT / "detector" / "models" / "face_landmarker.task"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, help="varsayilan: goruntunun yaninda .landmarks.json")
    args = parser.parse_args()

    mp = import_mediapipe()
    import cv2

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"goruntu okunamadi: {args.image}")
    height, width = image.shape[:2]
    print(f"goruntu: {width}x{height}")

    landmarker = build_landmarker(mp, args.model, "IMAGE")
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise SystemExit(
            "Render'da yuz BULUNAMADI. Muhtemel sebepler:\n"
            "  - kafa cerceveye sigmiyorr ya da cok kucuk\n"
            "  - render cok karanlik (07_render_head.py isik ekliyor ama sahneye bagli)\n"
            "  - materyal yok, yuz duz gri ve ozellik yok\n"
            f"Render'a bak: {args.image}"
        )

    points = result.face_landmarks[0]
    print(f"landmark: {len(points)}")

    out = args.out or args.image.with_suffix(".landmarks.json")
    out.write_text(
        json.dumps(
            {
                "image": str(args.image),
                "resolution": [width, height],
                # normalize [0,1]; piksel = x * width, y * height
                "landmarks": [
                    {"x": round(p.x, 6), "y": round(p.y, 6), "z": round(p.z, 6)} for p in points
                ],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"-> {out}")
    print("\nsonraki adim:")
    print(f"  blender --background <sahne.blend> --python scripts/09_build_correspondence.py -- {out}")


if __name__ == "__main__":
    main()
