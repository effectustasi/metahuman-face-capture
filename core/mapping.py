"""ARKit skorlarini face board GUI eksen degerlerine cevirir.

Sadece stdlib kullanir: bu modul Blender'in python'undan da, detector
venv'inden de, pytest'ten de ayni sekilde import edilir. Esleme mantigi
tek yerde durur, kopyalanmaz.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_MAPPING = Path(__file__).resolve().parent.parent / "mapping" / "arkit_to_mh.json"


class Mapping:
    """arkit_to_mh.json'un calisir hali.

    Tablo {arkit adi: [{gui, axis, weight}, ...]} seklinde. Ayni GUI eksenine
    birden fazla ARKit kanali dusebilir; katkilar TOPLANIR sonra kirpilir
    (Unreal'in davranisi). Toplama dogru olan: eyeBlinkLeft ve eyeWideLeft
    ayni eksenin arti/eksi yarisi, toplandiklarinda dogal olarak birbirini
    goturuyorlar -- max alsaydik ikisi de acikken yanlis sonuc verirdi.
    """

    def __init__(self, document: dict):
        self.document = document
        self.table: dict[str, list[dict]] = document["arkit"]
        self.limits: dict[str, dict[str, float]] = document.get("guiLimits", {})
        # tabloda yoklugu bilinen ve kabul edilen kanallar (bkz. docs/api-notes.md §8)
        self.known_unmapped: set[str] = set(document.get("knownUnmapped", []))
        self._axes = sorted({f"{c['gui']}.{c['axis']}" for e in self.table.values() for c in e})

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Mapping":
        path = Path(path) if path else DEFAULT_MAPPING
        return cls(json.loads(path.read_text(encoding="utf-8")))

    @property
    def axes(self) -> list[str]:
        """Tabloya gore surulen tum GUI eksenleri ('CTRL_C_jaw.y' formatinda)."""
        return list(self._axes)

    @property
    def bones(self) -> list[str]:
        """Dokunulan face board kemik isimleri."""
        return sorted({a.rsplit(".", 1)[0] for a in self._axes})

    def unknown_shapes(self, blendshapes: dict[str, float]) -> list[str]:
        """Girdide olup tabloda karsiligi olmayan BEKLENMEDIK isimler.

        Detector ile tablo arasinda isim kaymasi olursa sessizce yutulmasin
        diye. Bilinen bosluklar (knownUnmapped, ornegin mouthClose) haric --
        onlar her karede calarsa gercek hatayi gizler.
        """
        return sorted(
            name
            for name in blendshapes
            if name not in self.table and name != "_neutral" and name not in self.known_unmapped
        )

    def apply(
        self,
        blendshapes: dict[str, float],
        only: set[str] | None = None,
    ) -> dict[str, float]:
        """ARKit skorlari -> {'<kemik>.<eksen>': deger}.

        only: sadece bu ARKit kanallarini uygula. Tek kanalla dogrulama
        yaparken (once jawOpen) gerisini kapatmak icin.
        """
        accumulated: dict[str, float] = {}

        for name, score in blendshapes.items():
            if name == "_neutral":
                continue
            if only is not None and name not in only:
                continue
            entries = self.table.get(name)
            if not entries:
                continue
            for entry in entries:
                axis_key = f"{entry['gui']}.{entry['axis']}"
                accumulated[axis_key] = accumulated.get(axis_key, 0.0) + score * entry["weight"]

        # kirpma: DNA'nin kendi from/to araligi disina cikma
        for axis_key, value in accumulated.items():
            limit = self.limits.get(axis_key)
            if limit:
                accumulated[axis_key] = max(limit["min"], min(limit["max"], value))

        return accumulated


def split_axis_key(axis_key: str) -> tuple[str, int]:
    """'CTRL_C_jaw.y' -> ('CTRL_C_jaw', 1). Indeks Blender'in
    pose_bone.location vektor sirasidir: x=0, y=1, z=2."""
    bone, axis = axis_key.rsplit(".", 1)
    return bone, "xyz".index(axis)
