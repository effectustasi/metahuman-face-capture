"""charface'i Blender'a kurar -- kopyalayarak degil, BAGLAYARAK.

Blender'in scripts/addons klasorune kucuk bir yukleyici dosya yazar; gercek
kod projede kalir. Boylece kaynagi duzenleyip Blender'i yeniden baslatmak
yeterli olur, her degisiklikte yeniden kurmak gerekmez.

    python scripts/install_addon.py --version 5.0
    python scripts/install_addon.py --version 5.0 --uninstall

Kurduktan sonra Blender'da: Edit > Preferences > Add-ons > "charface" -> etkinlestir.
(Etkinlestirmeyi de otomatik yapmak icin scripts/enable_addon.py)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOADER_NAME = "charface_loader.py"

# Sablonun docstring'i RAW olmali: Windows yolundaki "C:\\Users" dizisi normal
# string'de gecersiz unicode kacisi olarak okunuyor ve modul import edilemiyor.
LOADER_TEMPLATE = '''r"""charface yukleyicisi -- otomatik uretildi, elle duzenleme.

Gercek kod: {addon_parent}\\charface
Yeniden uretmek icin: python scripts/install_addon.py --version <blender surumu>
"""

bl_info = {{
    "name": "charface",
    "description": "ARKit yuz yakalama -> MetaHuman face board",
    "author": "facecap",
    "version": (0, 1, 0),
    "blender": (4, 5, 0),
    "category": "Animation",
}}

import importlib
import sys
from pathlib import Path

ADDON_PARENT = Path(r"{addon_parent}")
PROJECT_ROOT = Path(r"{project_root}")

for path in (str(ADDON_PARENT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _fresh_import():
    """Her etkinlestirmede kaynaktan yeniden yukle. Gelistirirken
    Blender'i yeniden baslatmadan degisiklik gormek icin."""
    for name in ("core.mapping", "core.filters", "core.calibration", "core.take", "charface"):
        module = sys.modules.get(name)
        if module is not None:
            importlib.reload(module)
    import charface

    return charface


def register():
    _fresh_import().register()


def unregister():
    import charface

    charface.unregister()
'''


def blender_addons_directory(version: str) -> Path:
    if os.name == "nt":
        base = Path(os.environ["APPDATA"]) / "Blender Foundation" / "Blender"
    elif os.uname().sysname == "Darwin":  # type: ignore[attr-defined]
        base = Path.home() / "Library" / "Application Support" / "Blender"
    else:
        base = Path.home() / ".config" / "blender"
    return base / version / "scripts" / "addons"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="Blender surumu, ornek: 5.0")
    parser.add_argument("--uninstall", action="store_true")
    args = parser.parse_args()

    addons_directory = blender_addons_directory(args.version)
    loader_path = addons_directory / LOADER_NAME

    if args.uninstall:
        if loader_path.exists():
            loader_path.unlink()
            print(f"kaldirildi: {loader_path}")
        else:
            print("zaten kurulu degil")
        return

    if not addons_directory.parent.parent.exists():
        raise SystemExit(
            f"Blender {args.version} kullanici klasoru yok: {addons_directory.parent.parent}\n"
            "Surum numarasini kontrol et."
        )

    addons_directory.mkdir(parents=True, exist_ok=True)
    loader_path.write_text(
        LOADER_TEMPLATE.format(
            addon_parent=PROJECT_ROOT / "blender_addon",
            project_root=PROJECT_ROOT,
        ),
        encoding="utf-8",
    )
    print(f"kuruldu: {loader_path}")
    print(f"  -> kaynak: {PROJECT_ROOT / 'blender_addon' / 'charface'}")


if __name__ == "__main__":
    main()
