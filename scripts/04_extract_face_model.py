"""DNA'dan cozucu icin ileri modeli cikarir.

Neden gerekli: RigLogic joint ve blendshape **ciktisi** veriyor, vertex
vermiyor. Bir cozucunun gozlemle karsilastirabilmesi icin vertex konumu
lazim. Zinciri kendimiz kurmaliyiz:

    notr vertex
      + blendshape deltalari (RigLogic'in verdigi agirliklarla)
      -> linear blend skinning (RigLogic'in verdigi joint transformlariyla)
      = son vertex konumu

24049 vertexin hepsini her iterasyonda hesaplamak gereksiz; cozucu sadece
gozlemle esleyebildigi bolgeye bakar. Bu yuzden script bir VERTEX ALT
KUMESI cikariyor -- bolge, joint isimlerinden veri odakli seciliyor
(sahneye ihtiyac yok).

    blender --background --python scripts/04_extract_face_model.py -- \\
        <head.dna> <cikti.npz> [--region eyelid] [--threshold 0.5]
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import addon_utils
import numpy as np

ADDON_NAMES = ("character_dna_pro", "character_dna", "meta_human_dna")

# Bolge tanimlari: joint adinda gecen kaliplar. Vertexler bu jointlerin
# notr konumlarina yakinliklarina gore seciliyor.
REGIONS = {
    "eyelid": ("Eyelid", "EyeCorner", "Eyelash"),
    "eye": ("Eyelid", "EyeCorner", "Eyelash", "Eye"),
    "mouth": ("Lip", "Mouth", "Jaw"),
    "brow": ("Brow", "Forehead"),
    "all": (),
}


def find_addon() -> str:
    for module in addon_utils.modules():
        if module.__name__.split(".")[-1] in ADDON_NAMES:
            return module.__name__
    raise SystemExit(f"DNA addon bulunamadi ({ADDON_NAMES})")


def get_binding(module: str):
    bindings = importlib.import_module(f"{module}.bindings")
    for attribute in ("dna", "riglogic"):
        binding = getattr(bindings, attribute, None)
        if binding is not None and hasattr(binding, "FileStream"):
            return binding
    raise SystemExit("binding bulunamadi")


def enum_value(binding, flat: str, nested: str):
    value = getattr(binding, flat, None)
    if value is not None:
        return value
    holder, _, member = nested.partition(".")
    return getattr(getattr(binding, holder), member)


def open_reader(binding, path: Path):
    stream = binding.FileStream.create(
        path=str(path),
        accessMode=enum_value(binding, "AccessMode_Read", "AccessMode.Read"),
        openMode=enum_value(binding, "OpenMode_Binary", "OpenMode.Binary"),
        memRes=None,
    )
    reader = binding.BinaryStreamReader.create(
        stream,
        enum_value(binding, "DataLayer_All", "DataLayer.All"),
        enum_value(binding, "UnknownLayerPolicy_Preserve", "UnknownLayerPolicy.Preserve"),
        0,
        None,
    )
    reader.read()
    return reader


def select_vertices(reader, mesh_index: int, region: str, threshold: float):
    """Bolgeyi SKIN AGIRLIKLARINDAN sec, geometriden degil.

    Ilk denemede jointlerin notr dunya konumlarina yakinliga bakmistim;
    calismadi cunku DNA jointleri ebeveyne gore tutuyor ve dogru dunya
    konumu icin ebeveyn ROTASYONLARININ da zincirlenmesi gerekiyor --
    sadece oteleme toplamak konumlari kaydiriyor.

    Skin agirligi bu sorunu tamamen atlatiyor ve daha dogru bir tanim:
    bir vertex, agirliginin buyuk kismi goz kapagi jointlerine gidiyorsa
    goz kapagindadir. Geometrik yakinlik degil, rig'in kendi tanimi.
    """
    neutral = np.stack(
        [
            np.asarray(reader.getVertexPositionXs(mesh_index), dtype=np.float64),
            np.asarray(reader.getVertexPositionYs(mesh_index), dtype=np.float64),
            np.asarray(reader.getVertexPositionZs(mesh_index), dtype=np.float64),
        ],
        axis=1,
    )
    patterns = REGIONS[region]
    if not patterns:
        return np.arange(len(neutral)), neutral

    names = [reader.getJointName(i) for i in range(reader.getJointCount())]
    region_joints = {
        i for i, name in enumerate(names) if any(p.lower() in name.lower() for p in patterns)
    }
    if not region_joints:
        raise SystemExit(f"'{region}' bolgesi icin joint bulunamadi")

    selected = []
    for vertex in range(len(neutral)):
        joints = reader.getSkinWeightsJointIndices(mesh_index, vertex)
        weights = reader.getSkinWeightsValues(mesh_index, vertex)
        total = sum(
            float(w) for j, w in zip(joints, weights) if int(j) in region_joints
        )
        if total >= threshold:
            selected.append(vertex)

    selected = np.asarray(selected, dtype=np.int64)
    print(
        f"  bolge '{region}': {len(region_joints)} joint, esik {threshold} -> {len(selected)} vertex"
    )
    return selected, neutral


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(argv) < 2:
        raise SystemExit("kullanim: ... -- <head.dna> <cikti.npz> [--region R] [--threshold N] [--mesh N]")

    dna_path, out_path = Path(argv[0]), Path(argv[1])
    region = argv[argv.index("--region") + 1] if "--region" in argv else "eyelid"
    vertex_file = argv[argv.index("--vertices") + 1] if "--vertices" in argv else None
    threshold = float(argv[argv.index("--threshold") + 1]) if "--threshold" in argv else 0.5
    mesh_index = int(argv[argv.index("--mesh") + 1]) if "--mesh" in argv else 0

    binding = get_binding(find_addon())
    reader = open_reader(binding, dna_path)
    print(f"DNA: GUI={reader.getGUIControlCount()} RAW={reader.getRawControlCount()} joint={reader.getJointCount()}")
    print(f"mesh[{mesh_index}] '{reader.getMeshName(mesh_index)}' vertex={reader.getVertexPositionCount(mesh_index)}")

    if vertex_file:
        # Karsilik dosyasindaki vertexleri kullan -- cozucu tam olarak
        # onlari gozlemle karsilastiracak, baskasini hesaplamak bosuna.
        import json

        payload = json.loads(Path(vertex_file).read_text(encoding="utf-8"))
        wanted = sorted({int(v) for r in payload["correspondence"] for v in r["vertices"]})
        neutral_all = np.stack(
            [
                np.asarray(reader.getVertexPositionXs(mesh_index), dtype=np.float64),
                np.asarray(reader.getVertexPositionYs(mesh_index), dtype=np.float64),
                np.asarray(reader.getVertexPositionZs(mesh_index), dtype=np.float64),
            ],
            axis=1,
        )
        selected = np.asarray(wanted, dtype=np.int64)
        region = f"correspondence:{Path(vertex_file).stem}"
        print(f"  karsilik dosyasindan {len(selected)} vertex")
    else:
        selected, neutral_all = select_vertices(reader, mesh_index, region, threshold)
    neutral = neutral_all[selected]
    lookup = {int(v): i for i, v in enumerate(selected)}

    # --- skin agirliklari (sadece secilen vertexler)
    max_influences = 0
    weight_rows, joint_rows = [], []
    for vertex in selected:
        weights = np.asarray(reader.getSkinWeightsValues(mesh_index, int(vertex)), dtype=np.float64)
        joints = np.asarray(reader.getSkinWeightsJointIndices(mesh_index, int(vertex)), dtype=np.int32)
        weight_rows.append(weights)
        joint_rows.append(joints)
        max_influences = max(max_influences, len(weights))

    skin_weights = np.zeros((len(selected), max_influences), dtype=np.float64)
    skin_joints = np.zeros((len(selected), max_influences), dtype=np.int32)
    for row, (weights, joints) in enumerate(zip(weight_rows, joint_rows)):
        skin_weights[row, : len(weights)] = weights
        skin_joints[row, : len(joints)] = joints
    print(f"  skin: vertex basina en fazla {max_influences} joint")

    # --- blendshape deltalari (sadece secilen vertexlere dokunanlar)
    target_count = reader.getBlendShapeTargetCount(mesh_index)
    deltas, channels = [], []
    for target in range(target_count):
        indices = np.asarray(reader.getBlendShapeTargetVertexIndices(mesh_index, target), dtype=np.int32)
        hit = [(lookup[int(v)], k) for k, v in enumerate(indices) if int(v) in lookup]
        if not hit:
            continue
        dx = np.asarray(reader.getBlendShapeTargetDeltaXs(mesh_index, target), dtype=np.float64)
        dy = np.asarray(reader.getBlendShapeTargetDeltaYs(mesh_index, target), dtype=np.float64)
        dz = np.asarray(reader.getBlendShapeTargetDeltaZs(mesh_index, target), dtype=np.float64)

        block = np.zeros((len(selected), 3), dtype=np.float64)
        for row, k in hit:
            block[row] = (dx[k], dy[k], dz[k])
        deltas.append(block)
        channels.append(target)

    delta_stack = np.stack(deltas) if deltas else np.zeros((0, len(selected), 3))
    print(f"  blendshape: {target_count} hedeften {len(channels)} tanesi bu bolgeye dokunuyor")

    # --- joint notr transformlari (skinning icin)
    neutral_translation = np.stack(
        [
            np.asarray(reader.getNeutralJointTranslationXs(), dtype=np.float64),
            np.asarray(reader.getNeutralJointTranslationYs(), dtype=np.float64),
            np.asarray(reader.getNeutralJointTranslationZs(), dtype=np.float64),
        ],
        axis=1,
    )
    neutral_rotation = np.stack(
        [
            np.asarray(reader.getNeutralJointRotationXs(), dtype=np.float64),
            np.asarray(reader.getNeutralJointRotationYs(), dtype=np.float64),
            np.asarray(reader.getNeutralJointRotationZs(), dtype=np.float64),
        ],
        axis=1,
    )
    parents = np.asarray(
        [reader.getJointParentIndex(i) for i in range(reader.getJointCount())], dtype=np.int32
    )

    gui_names = [reader.getGUIControlName(i) for i in range(reader.getGUIControlCount())]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        region=region,
        meshIndex=mesh_index,
        vertexIndices=selected.astype(np.int32),
        neutral=neutral,
        skinWeights=skin_weights,
        skinJoints=skin_joints,
        blendShapeDeltas=delta_stack,
        blendShapeChannels=np.asarray(channels, dtype=np.int32),
        jointNeutralTranslation=neutral_translation,
        jointNeutralRotation=neutral_rotation,
        jointParents=parents,
        guiControlNames=np.asarray(gui_names),
        sourceDna=str(dna_path),
    )
    size = out_path.stat().st_size / 1024
    print(f"\n{len(selected)} vertex, {len(channels)} blendshape -> {out_path} ({size:.0f} KB)")


main()
