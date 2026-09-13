"""Sahnedeki Character DNA kurulumunu dok: addon adi, rig instance, face board.

Kod yazmadan once ve addon surumu degistiginde calistirilir. Amac API
isimlerini EZBERDEN yazmamak -- neyin gercekten var oldugunu buradan
dogrula.

    blender --background <sahne.blend> --python scripts/00_introspect.py -- cikti.json

Sahne vermeden de calisir (o zaman sadece addon/kurulum bilgisi doker):

    blender --background --python scripts/00_introspect.py -- cikti.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import addon_utils
import bpy

ADDON_SUFFIX = "character_dna"  # 'character_dna' ve 'character_dna_pro' ikisini de yakalar


def describe_addons() -> dict:
    enabled = {a.module for a in bpy.context.preferences.addons}
    modules = [m.__name__ for m in addon_utils.modules() if ADDON_SUFFIX in m.__name__.lower()]
    return {
        "matching": modules,
        "enabled": sorted(m for m in modules if m in enabled),
        "extensionRepos": [
            {"module": r.module, "directory": r.directory}
            for r in bpy.context.preferences.extensions.repos
        ],
        "blenderVersion": bpy.app.version_string,
        "pythonVersion": sys.version.split()[0],
    }


def describe_scene_properties() -> dict:
    """Addon'un sahneye hangi isimle bagli oldugunu bul."""
    out = {}
    for name in dir(bpy.types.Scene):
        if ADDON_SUFFIX in name.lower():
            group = getattr(bpy.context.scene, name, None)
            out[name] = {
                "properties": sorted(
                    p.identifier for p in group.bl_rna.properties if p.identifier != "rna_type"
                )
                if group is not None
                else None
            }
    return out


def describe_rig_instances() -> list:
    instances = []
    for scene_property in describe_scene_properties():
        group = getattr(bpy.context.scene, scene_property, None)
        rig_list = getattr(group, "rig_instance_list", None)
        if not rig_list:
            continue
        for instance in rig_list:
            face_board = getattr(instance, "face_board", None)
            instances.append(
                {
                    "sceneProperty": scene_property,
                    "name": getattr(instance, "name", None),
                    "methods": sorted(
                        m for m in dir(instance) if not m.startswith("_") and callable(getattr(instance, m, None))
                    ),
                    "faceBoard": {
                        "object": face_board.name if face_board else None,
                        "boneCount": len(face_board.pose.bones) if face_board else 0,
                        "bones": [b.name for b in face_board.pose.bones] if face_board else [],
                    },
                    "headRig": getattr(getattr(instance, "head_rig", None), "name", None),
                    "headMesh": getattr(getattr(instance, "head_mesh", None), "name", None),
                }
            )
    return instances


def describe_armatures() -> dict:
    return {
        obj.name: {
            "boneCount": len(obj.pose.bones),
            "bones": [b.name for b in obj.pose.bones][:400],
            "customProperties": {k: str(v) for k, v in obj.items()},
        }
        for obj in bpy.data.objects
        if obj.type == "ARMATURE"
    }


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    out_path = Path(argv[0]) if argv else Path("introspect.json")

    report = {
        "addons": describe_addons(),
        "sceneProperties": describe_scene_properties(),
        "rigInstances": describe_rig_instances(),
        "armatures": describe_armatures(),
        "operators": sorted(o for o in dir(bpy.ops) if ADDON_SUFFIX in o.lower() or "charface" in o.lower()),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"addon        : {report['addons']['enabled'] or report['addons']['matching']}")
    print(f"sahne prop   : {list(report['sceneProperties'])}")
    print(f"rig instance : {len(report['rigInstances'])}")
    print(f"armature     : {list(report['armatures'])}")
    print(f"-> {out_path}")


main()
