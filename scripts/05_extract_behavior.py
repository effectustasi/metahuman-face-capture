"""RigLogic'in davranis katmanini disari cikarir (turevlenebilir yeniden yazim icin).

Zincir -- API yoklanarak cikarildi, ezberden degil:

    GUI (174)      kullanicinin/face board'un surdugu degerler
      |  parcali dogrusal: raw = slope * gui + cut,  gui in [from, to]
    raw (263)
      |  PSD: ham kontrollerin agirlikli CARPIMLARI, raw'in ardina eklenir
    kontrol (808 = 263 + 545)
      |  joint gruplari: 122 yogun matris, toplam 1.000.930 katsayi
    joint deltalari (7830 = 870 joint x 9)
      |  blendshape kanallari: birebir gather
    blendshape agirliklari (782)

Hepsi turevlenebilir. Joint gruplari saf matmul -- GPU'da batch'lenebilir.

    blender --background --python scripts/05_extract_behavior.py -- <head.dna> <cikti.npz>
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import addon_utils
import numpy as np

ADDON_NAMES = ("character_dna_pro", "character_dna", "meta_human_dna")


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


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(argv) < 2:
        raise SystemExit("kullanim: ... -- <head.dna> <cikti.npz>")
    dna_path, out_path = Path(argv[0]), Path(argv[1])

    reader = open_reader(get_binding(find_addon()), dna_path)

    gui_count = reader.getGUIControlCount()
    raw_count = reader.getRawControlCount()
    psd_count = reader.getPSDCount()
    joint_count = reader.getJointCount()
    blend_count = reader.getBlendShapeChannelCount()
    # Kontrol vektoru sadece raw+PSD degil: RBF poz kontrolleri de sonuna
    # ekleniyor. Olculdu -- joint grubu girdi indeksleri 825'e kadar gidiyor,
    # raw+PSD ise 808'de bitiyor.
    rbf_count = reader.getRBFPoseControlCount() if hasattr(reader, "getRBFPoseControlCount") else 0
    control_count = raw_count + psd_count + rbf_count
    print(f"GUI={gui_count} RAW={raw_count} PSD={psd_count} RBF={rbf_count} -> kontrol={control_count}")
    print(f"joint={joint_count} blendshape={blend_count}")

    payload = {
        "guiCount": gui_count,
        "rawCount": raw_count,
        "psdCount": psd_count,
        "rbfPoseControlCount": rbf_count,
        "controlCount": control_count,
        "jointCount": joint_count,
        "blendShapeChannelCount": blend_count,
        "guiControlNames": np.asarray([reader.getGUIControlName(i) for i in range(gui_count)]),
        "rawControlNames": np.asarray([reader.getRawControlName(i) for i in range(raw_count)]),
        "sourceDna": str(dna_path),
    }

    # --- 1) GUI -> raw parcali dogrusal
    payload["guiToRawInputIndices"] = np.asarray(reader.getGUIToRawInputIndices(), dtype=np.int32)
    payload["guiToRawOutputIndices"] = np.asarray(reader.getGUIToRawOutputIndices(), dtype=np.int32)
    payload["guiToRawFromValues"] = np.asarray(reader.getGUIToRawFromValues(), dtype=np.float32)
    payload["guiToRawToValues"] = np.asarray(reader.getGUIToRawToValues(), dtype=np.float32)
    payload["guiToRawSlopeValues"] = np.asarray(reader.getGUIToRawSlopeValues(), dtype=np.float32)
    payload["guiToRawCutValues"] = np.asarray(reader.getGUIToRawCutValues(), dtype=np.float32)
    print(f"  GUI->raw satiri: {len(payload['guiToRawInputIndices'])}")

    # --- 2) PSD: seyrek carpim tablosu
    # row = cikti indeksi (raw'in ardindan, yani >= rawCount)
    # column = carpima giren kontrol indeksi
    payload["psdRowIndices"] = np.asarray(reader.getPSDRowIndices(), dtype=np.int32)
    payload["psdColumnIndices"] = np.asarray(reader.getPSDColumnIndices(), dtype=np.int32)
    payload["psdValues"] = np.asarray(reader.getPSDValues(), dtype=np.float32)
    print(f"  PSD girdisi: {len(payload['psdRowIndices'])}")

    # --- 3) joint gruplari: her biri yogun (cikti x girdi) matris
    group_count = reader.getJointGroupCount()
    inputs, outputs, values, offsets = [], [], [], [0]
    shapes = []
    for group in range(group_count):
        group_inputs = np.asarray(reader.getJointGroupInputIndices(group), dtype=np.int32)
        group_outputs = np.asarray(reader.getJointGroupOutputIndices(group), dtype=np.int32)
        group_values = np.asarray(reader.getJointGroupValues(group), dtype=np.float32)
        if not len(group_inputs) or not len(group_outputs):
            shapes.append((0, 0))
            offsets.append(offsets[-1])
            continue
        expected = len(group_outputs) * len(group_inputs)
        if len(group_values) != expected:
            raise SystemExit(
                f"grup {group}: {len(group_values)} katsayi, {expected} bekleniyordu"
            )
        inputs.append(group_inputs)
        outputs.append(group_outputs)
        values.append(group_values)
        shapes.append((len(group_outputs), len(group_inputs)))
        offsets.append(offsets[-1] + len(group_values))

    payload["jointGroupInputIndices"] = np.concatenate(inputs) if inputs else np.zeros(0, np.int32)
    payload["jointGroupOutputIndices"] = np.concatenate(outputs) if outputs else np.zeros(0, np.int32)
    payload["jointGroupValues"] = np.concatenate(values) if values else np.zeros(0, np.float32)
    payload["jointGroupShapes"] = np.asarray(shapes, dtype=np.int32)
    payload["jointGroupValueOffsets"] = np.asarray(offsets, dtype=np.int64)
    # girdi/cikti dizileri de gruplara bolunebilsin
    payload["jointGroupInputOffsets"] = np.cumsum(
        [0] + [s[1] for s in shapes], dtype=np.int64
    )
    payload["jointGroupOutputOffsets"] = np.cumsum(
        [0] + [s[0] for s in shapes], dtype=np.int64
    )
    print(f"  joint grubu: {group_count}, toplam katsayi {len(payload['jointGroupValues'])}")

    # --- 4) blendshape kanal eslemesi
    payload["blendShapeInputIndices"] = np.asarray(
        reader.getBlendShapeChannelInputIndices(), dtype=np.int32
    )
    payload["blendShapeOutputIndices"] = np.asarray(
        reader.getBlendShapeChannelOutputIndices(), dtype=np.int32
    )
    print(f"  blendshape eslemesi: {len(payload['blendShapeInputIndices'])}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    print(f"\n-> {out_path} ({out_path.stat().st_size/1024/1024:.1f} MB)")


main()
