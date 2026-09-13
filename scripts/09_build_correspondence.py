"""Landmark <-> vertex karsiligini kurar (isin atarak).

Karsilik zincirinin 3. adimi:
  1. Panelde "Kafayi Render Et"  -> render + kamera (Blender)
  2. scripts/08_landmarks.py     -> 478 landmark (detector venv)
  3. **bu script**               -> her landmark'tan isin at, mesh'e carptigi
                                    ucgeni ve baryantrik agirliklari bul

Neden baryantrik, en yakin vertex degil: landmark bir vertexin uzerine
tam oturmuyor, ucgenin ortasina dusuyor. En yakin vertexi almak yari
vertex araligi kadar hata demek (bu mesh'te ~1 mm). Uc vertexin agirlikli
ortalamasi hem daha dogru hem de cozucu icin turevlenebilir.

    blender --background <sahne.blend> --python scripts/09_build_correspondence.py -- \\
        [<landmarks.json>] [<cikti.json>]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from mathutils.geometry import barycentric_transform, intersect_point_tri

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = PROJECT_ROOT / "mapping" / "_generated" / "correspondence"


def camera_from_json(data: dict):
    """Render'da kullanilan kamerayi yeniden kur.

    Render sirasinda kamera gecici olusturulup silindi (sahne kirlenmesin
    diye); isin atmak icin ayni parametrelerle geri kuruyoruz.
    """
    camera_data = bpy.data.cameras.new("charface_corr_cam")
    camera = bpy.data.objects.new("charface_corr_cam", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    camera.location = Vector(data["location"])
    camera.rotation_euler = data["rotationEuler"]
    camera_data.lens = data["lens"]
    camera_data.sensor_width = data["sensorWidth"]
    bpy.context.view_layer.update()
    return camera, camera_data


def ray_for_pixel(camera, scene, u: float, v: float):
    """Normalize goruntu koordinatindan (u sag, v ASAGI) dunya isini.

    MediaPipe y'yi yukaridan asagi veriyor, Blender'in view_frame'i
    asagidan yukari -- ceviriyoruz.
    """
    frame = camera.data.view_frame(scene=scene)  # sag-ust, sag-alt, sol-alt, sol-ust
    top_right, bottom_right, bottom_left, top_left = frame

    top = top_left.lerp(top_right, u)
    bottom = bottom_left.lerp(bottom_right, u)
    point_local = top.lerp(bottom, v)  # v=0 ust, v=1 alt

    origin = camera.matrix_world.translation
    target = camera.matrix_world @ point_local
    return origin, (target - origin).normalized()


def polygon_triangles(mesh, polygon):
    """Cokgeni ucgen yelpazesine bol. MetaHuman mesh'i dortgen."""
    indices = list(polygon.vertices)
    for k in range(1, len(indices) - 1):
        yield indices[0], indices[k], indices[k + 1]


def barycentric_weights(mesh, matrix, polygon, hit_world):
    """Carpma noktasini iceren ucgeni bul, baryantrik agirliklari dondur."""
    for a, b, c in polygon_triangles(mesh, polygon):
        pa = matrix @ mesh.vertices[a].co
        pb = matrix @ mesh.vertices[b].co
        pc = matrix @ mesh.vertices[c].co
        if intersect_point_tri(hit_world, pa, pb, pc) is None:
            continue
        weights = barycentric_transform(
            hit_world, pa, pb, pc, Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))
        )
        return [a, b, c], [round(w, 6) for w in weights]

    # Hicbir ucgen icermiyorsa (kenar durumu) en yakin vertexe dus
    best = min(polygon.vertices, key=lambda i: (matrix @ mesh.vertices[i].co - hit_world).length)
    return [int(best), int(best), int(best)], [1.0, 0.0, 0.0]


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    landmarks_path = Path(argv[0]) if argv else DEFAULT_DIR / "head_front.landmarks.json"
    out_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_DIR / "correspondence.json"
    camera_path = landmarks_path.parent / "camera.json"

    for path in (landmarks_path, camera_path):
        if not path.exists():
            raise SystemExit(f"eksik dosya: {path}")

    landmark_data = json.loads(landmarks_path.read_text(encoding="utf-8"))
    camera_info = json.loads(camera_path.read_text(encoding="utf-8"))

    mesh_object = bpy.data.objects.get(camera_info["meshObject"])
    if mesh_object is None:
        raise SystemExit(f"mesh bulunamadi: {camera_info['meshObject']}")

    scene = bpy.context.scene
    camera, camera_data = camera_from_json(camera_info)

    try:
        # Isin atmayi DEGERLENDIRILMIS mesh uzerinde yapmiyoruz: render notr
        # pozda alindi, karsilik da notr pozda kurulmali.
        mesh = mesh_object.data
        matrix = mesh_object.matrix_world
        inverse = matrix.inverted()

        records = []
        missed = []
        for index, point in enumerate(landmark_data["landmarks"]):
            u, v = point["x"], point["y"]
            if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
                missed.append({"index": index, "reason": "kare disinda", "u": u, "v": v})
                continue

            origin, direction = ray_for_pixel(camera, scene, u, v)
            hit, location, _, face_index = mesh_object.ray_cast(
                inverse @ origin, inverse.to_3x3() @ direction
            )
            if not hit:
                missed.append({"index": index, "reason": "mesh'e carpmadi", "u": u, "v": v})
                continue

            hit_world = matrix @ location
            vertices, weights = barycentric_weights(
                mesh, matrix, mesh.polygons[face_index], hit_world
            )
            records.append(
                {
                    "landmark": index,
                    "vertices": [int(v) for v in vertices],
                    "weights": weights,
                    "position": [round(c, 6) for c in hit_world],
                }
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "meshObject": mesh_object.name,
                    "landmarkCount": len(landmark_data["landmarks"]),
                    "resolved": len(records),
                    "missed": missed,
                    "correspondence": records,
                },
                indent=1,
            ),
            encoding="utf-8",
        )

        print(f"cozulen : {len(records)}/{len(landmark_data['landmarks'])}")
        if missed:
            print(f"kacan   : {len(missed)}")
            for item in missed[:5]:
                print(f"   landmark {item['index']}: {item['reason']} (u={item['u']:.3f} v={item['v']:.3f})")
        print(f"-> {out_path}")
    finally:
        bpy.data.objects.remove(camera, do_unlink=True)
        bpy.data.cameras.remove(camera_data, do_unlink=True)


main()
