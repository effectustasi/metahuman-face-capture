"""DNA'dan GUI/raw kontrol isimlerini ve GUI->raw eslemesini disari dokum.

Blender icinde calisir (bindings derlenmis .pyd, sadece Blender'in python'unda yuklenir):

    blender --background --python facecap/scripts/01_dump_dna_controls.py -- <head.dna> <cikti.json>

Cikti: mapping/_generated/dna_controls.json
"""

import json
import sys
from pathlib import Path

ADDON_NAMES = ("character_dna_pro", "character_dna", "meta_human_dna")


def find_addon_module():
    """Addon modul yolunu ARAYARAK bul, sabit yazma.

    Extension repo adi makineden makineye degisiyor (burada 'polyhammer_com',
    diskte bayat bir 'polyhammer' kopyasi da var). addon_utils tek dogru kaynak.
    """
    import addon_utils

    names = [
        m.__name__
        for m in addon_utils.modules()
        if any(m.__name__.split(".")[-1] == n for n in ADDON_NAMES)
    ]
    # etkin olan varsa onu tercih et
    import bpy

    enabled = {a.module for a in bpy.context.preferences.addons}
    for name in names:
        if name in enabled:
            return name
    if names:
        return names[0]
    raise SystemExit(f"su addonlardan biri bulunamadi: {ADDON_NAMES}. Kurulu mu?")


def get_bindings():
    """Addon paketi uzerinden binding'i al. Dogrudan sys.path ile import
    calismaz: SWIG wrapper'lari paketin IsolatedModuleLoader hook'unu ariyor.

    Modul adi surume gore degisiyor:
      0.8.7 -> bindings.dna       (dna.OpenMode_Binary, dna.DataLayer_All)
      0.5.4 -> bindings.riglogic  (riglogic.OpenMode.Binary, riglogic.DataLayer.All)
    """
    import importlib

    module = find_addon_module()
    try:
        bindings = importlib.import_module(f"{module}.bindings")
    except Exception:
        import bpy

        bpy.ops.preferences.addon_enable(module=module)
        bindings = importlib.import_module(f"{module}.bindings")

    for attribute in ("dna", "riglogic"):
        binding = getattr(bindings, attribute, None)
        if binding is not None and hasattr(binding, "FileStream"):
            return binding
    raise SystemExit(f"{module}.bindings icinde dna/riglogic bulunamadi")


def enum_value(binding, flat_name: str, nested_path: str):
    """0.8.7 duz isim kullaniyor (dna.OpenMode_Binary), 0.5.4 ic ice
    (riglogic.OpenMode.Binary). Ikisini de dene."""
    value = getattr(binding, flat_name, None)
    if value is not None:
        return value
    holder_name, _, member = nested_path.partition(".")
    holder = getattr(binding, holder_name, None)
    if holder is not None:
        value = getattr(holder, member, None)
        if value is not None:
            return value
    raise SystemExit(f"binding icinde ne {flat_name} ne {nested_path} var")


def open_reader(binding, path):
    layer = "Behavior"
    if not (hasattr(binding, "DataLayer_Behavior") or hasattr(getattr(binding, "DataLayer", None), "Behavior")):
        layer = "All"

    stream = binding.FileStream.create(
        path=str(path),
        accessMode=enum_value(binding, "AccessMode_Read", "AccessMode.Read"),
        openMode=enum_value(binding, "OpenMode_Binary", "OpenMode.Binary"),
        memRes=None,
    )
    reader = binding.BinaryStreamReader.create(
        stream,
        enum_value(binding, f"DataLayer_{layer}", f"DataLayer.{layer}"),
        enum_value(binding, "UnknownLayerPolicy_Preserve", "UnknownLayerPolicy.Preserve"),
        0,
        None,
    )
    reader.read()
    return reader


def dump(reader):
    gui = [reader.getGUIControlName(i) for i in range(reader.getGUIControlCount())]
    raw = [reader.getRawControlName(i) for i in range(reader.getRawControlCount())]

    out = {
        "guiControlCount": len(gui),
        "rawControlCount": len(raw),
        "guiControlNames": gui,
        "rawControlNames": raw,
    }

    # GUI -> raw piecewise-linear eslemesi. Isim seti surumden surume degisebiliyor,
    # bu yuzden var olanlari tarayip ekliyoruz.
    for attr in (
        "getGUIToRawInputIndices",
        "getGUIToRawOutputIndices",
        "getGUIToRawFromValues",
        "getGUIToRawToValues",
        "getGUIToRawSlopeValues",
        "getGUIToRawCutValues",
    ):
        fn = getattr(reader, attr, None)
        if fn is None:
            out.setdefault("missing", []).append(attr)
            continue
        key = attr[3:]  # getGUIToRawFromValues -> GUIToRawFromValues
        key = key[0].lower() + key[1:] if not key.startswith("GUI") else "guiToRaw" + key[8:]
        # 0.5.4 binding'leri numpy skalari donduruyor, json serilestiremiyor.
        cast = int if key.endswith("Indices") else float
        out[key] = [cast(value) for value in fn()]

    return out


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(argv) < 2:
        raise SystemExit("kullanim: ... -- <head.dna> <cikti.json>")
    dna_path, out_path = Path(argv[0]), Path(argv[1])

    binding = get_bindings()
    reader = open_reader(binding, dna_path)
    data = dump(reader)
    data["sourceDna"] = str(dna_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"OK gui={data['guiControlCount']} raw={data['rawControlCount']} -> {out_path}")


main()
