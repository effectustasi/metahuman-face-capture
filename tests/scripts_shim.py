"""02_build_mapping.py'deki kanonik listeye testten erismek icin kopru.

Script adi rakamla basladigi icin dogrudan import edilemiyor.
"""

import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parent.parent / "scripts" / "02_build_mapping.py"
_spec = importlib.util.spec_from_file_location("build_mapping", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

MEDIAPIPE_BLENDSHAPES = _module.MEDIAPIPE_BLENDSHAPES
ARKIT_ONLY = _module.ARKIT_ONLY
AUTHORED = _module.AUTHORED
