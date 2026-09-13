"""Gercek RigLogic'ten referans cikti uretir.

`core/riglogic_torch.py` RigLogic'i yeniden yaziyor. Yeniden yazimin
dogrulugu ancak GERCEK kutuphaneye karsi kanitlanabilir: ayni GUI
girdisine ayni joint/blendshape ciktisi.

Torch Blender'in python'unda yok, RigLogic de torch venv'inde yok --
bu yuzden referans burada uretilip dosyaya yaziliyor, karsilastirma
testte yapiliyor.

    blender --background --python scripts/06_dump_reference.py -- <head.dna> <cikti.npz> [ornek]
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
    raise SystemExit("DNA addon bulunamadi")


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


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(argv) < 2:
        raise SystemExit("kullanim: ... -- <head.dna> <cikti.npz> [ornek sayisi]")
    dna_path, out_path = Path(argv[0]), Path(argv[1])
    samples = int(argv[2]) if len(argv) > 2 else 32

    binding = get_binding(find_addon())
    stream = binding.FileStream.create(
        path=str(dna_path),
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

    rig = binding.RigLogic.create(reader=reader, config=binding.Configuration(), memRes=None)
    instance = binding.RigInstance.create(rigLogic=rig, memRes=None)

    gui_count = reader.getGUIControlCount()
    print(f"GUI={gui_count} ornek={samples}")

    rng = np.random.default_rng(12345)
    gui_inputs, joint_outputs, blend_outputs = [], [], []

    for index in range(samples):
        if index == 0:
            gui = np.zeros(gui_count, dtype=np.float64)  # notr
        elif index == 1:
            gui = np.ones(gui_count, dtype=np.float64)  # hepsi acik
        elif index == 2:
            gui = -np.ones(gui_count, dtype=np.float64)  # hepsi negatif
        else:
            # cogu kontrol kapali, birkaci acik -- gercek kullanima benzer
            gui = np.zeros(gui_count, dtype=np.float64)
            active = rng.choice(gui_count, size=rng.integers(1, 12), replace=False)
            gui[active] = rng.uniform(-1.0, 1.0, size=len(active))

        for control in range(gui_count):
            instance.setGUIControl(control, float(gui[control]))
        rig.mapGUIToRawControls(instance)
        rig.calculate(instance)

        gui_inputs.append(gui)
        joint_outputs.append(np.asarray(instance.getRawJointOutputs(), dtype=np.float64))
        blend_outputs.append(np.asarray(instance.getBlendShapeOutputs(), dtype=np.float64))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        gui=np.stack(gui_inputs),
        jointOutputs=np.stack(joint_outputs),
        blendOutputs=np.stack(blend_outputs),
        sourceDna=str(dna_path),
    )
    print(f"joint cikti={joint_outputs[0].shape} blendshape cikti={blend_outputs[0].shape}")
    print(f"-> {out_path} ({out_path.stat().st_size/1024:.0f} KB)")


main()
