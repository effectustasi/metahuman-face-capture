"""Karakterin kafasini onden render eder (landmark karsiligi kurmak icin).

Cozucunun gozlemle karsilastirabilmesi icin sunu bilmesi lazim: 24049
vertexten hangisi MediaPipe'in hangi landmark'i?

Yaklasim: karakterin KENDI kafasini render et, MediaPipe'i o goruntuye
calistir, her landmark'tan kameraya dogru isin at, mesh'e carptigi yeri
bul. Boylece karsilik jenerik bir yuz modeliyle degil, bu DNA'yla kurulur
ve hizalama/ICP adimi hic olmaz.

MediaPipe Blender'da olmadigi icin is uc adima bolundu:

    1. bu script          -> render + kamera parametreleri (.png + .json)
    2. scripts/08_landmarks.py (detector venv) -> landmark'lar (.json)
    3. scripts/09_build_correspondence.py (Blender) -> isin at, vertex bul

    blender --background <sahne.blend> --python scripts/07_render_head.py -- <cikti klasoru>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT / "blender_addon"))


def main():
    """Headless kullanim. Panelden calistirmak icin: charface > Kafayi Render Et.

    Mantik addon'da (`charface.render_head_reference`) -- tek kaynak, boylece
    panel ve komut satiri ayni sekilde davraniyor.
    """
    import bpy

    from charface import render_head_reference

    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    out_dir = Path(argv[0]) if argv else PROJECT_ROOT / "mapping" / "_generated" / "correspondence"

    image_path, mesh_name = render_head_reference(bpy.context, out_dir)
    print(f"mesh: {mesh_name}")
    print(f"render -> {image_path}")
    print("\nsonraki adim:")
    print(f'  detector/.venv/Scripts/python scripts/08_landmarks.py "{image_path}"')


main()
