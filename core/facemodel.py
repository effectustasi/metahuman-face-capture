"""Rig'in ileri modeli: kontrol vektoru -> vertex konumlari.

Cozucunun kalbi. RigLogic joint ve blendshape **ciktisi** veriyor, vertex
vermiyor; gozlemle karsilastirmak icin zinciri kendimiz kuruyoruz:

    notr vertex
      + blendshape deltalari       (RigLogic'in verdigi agirliklarla)
      -> linear blend skinning     (RigLogic'in verdigi joint deltalariyla)
      = son vertex konumu

Semantik addon kaynagindan okundu, tahmin yok (`rig_logic.py:1613`):

* `getRawJointOutputs()` joint basina **9** deger verir
* [0:3] oteleme **deltasi**, SCALE_FACTOR=100'e bolunur
* [3:6] rotasyon **deltasi**, **derece**
* [6:9] olcek deltasi
* uygulama `notr + delta`, Euler sirasi **XYZ**

DNA'daki notr rotasyonlar da derece (`dna_io/importer.py:476`).

numpy kullanir. Blender numpy'i paketiyle birlikte getiriyor, detector
venv'inde de var; yani core'un geri kalanindaki "sadece stdlib" kuralini
bozmadan burada kullanilabiliyor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# DNA santimetre, Blender metre. constants.py:38
SCALE_FACTOR = 100.0
JOINT_STRIDE = 9  # oteleme(3) + rotasyon(3) + olcek(3)


def euler_xyz_to_matrix(angles: np.ndarray) -> np.ndarray:
    """(N, 3) radyan XYZ Euler -> (N, 3, 3) rotasyon matrisi.

    Sira addon'un kullandigiyla ayni olmali: Blender'in Euler('XYZ')
    bilesimi R = Rz @ Ry @ Rx seklinde uygular.
    """
    x, y, z = angles[:, 0], angles[:, 1], angles[:, 2]
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)

    count = len(angles)
    rotation = np.empty((count, 3, 3), dtype=np.float64)
    rotation[:, 0, 0] = cy * cz
    rotation[:, 0, 1] = cz * sx * sy - cx * sz
    rotation[:, 0, 2] = cx * cz * sy + sx * sz
    rotation[:, 1, 0] = cy * sz
    rotation[:, 1, 1] = cx * cz + sx * sy * sz
    rotation[:, 1, 2] = -cz * sx + cx * sy * sz
    rotation[:, 2, 0] = -sy
    rotation[:, 2, 1] = cy * sx
    rotation[:, 2, 2] = cx * cy
    return rotation


def compose(translation: np.ndarray, rotation: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """(N,3) oteleme + (N,3,3) rotasyon + (N,3) olcek -> (N,4,4)."""
    count = len(translation)
    matrix = np.zeros((count, 4, 4), dtype=np.float64)
    matrix[:, :3, :3] = rotation * scale[:, None, :]
    matrix[:, :3, 3] = translation
    matrix[:, 3, 3] = 1.0
    return matrix


def accumulate_world(local: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Hiyerarsiyi zincirle -- basit surum, referans olarak duruyor.

    Joint basina bir python adimi. 870 joint icin ~1.6 ms; hizli yol icin
    `build_levels` + `accumulate_world_levels` kullanilir.
    """
    world = np.empty_like(local)
    for index in range(len(local)):
        parent = int(parents[index])
        world[index] = local[index] if parent == index or parent < 0 else world[parent] @ local[index]
    return world


def build_levels(parents: np.ndarray) -> list[np.ndarray]:
    """Jointleri hiyerarsi derinligine gore grupla.

    Ayni seviyedeki jointlerin hepsi ayni anda hesaplanabilir, cunku
    hepsinin ebeveyni bir onceki seviyede bitmis olur. Boylece python
    dongusu joint sayisi kadar degil, DERINLIK kadar donuyor -- olculdu:
    goz kapagi bolgesinde 128 joint ama sadece 8 seviye.
    """
    depth = np.zeros(len(parents), dtype=np.int64)
    for index in range(len(parents)):
        parent = int(parents[index])
        depth[index] = 0 if (parent == index or parent < 0) else depth[parent] + 1

    return [np.nonzero(depth == level)[0] for level in range(int(depth.max()) + 1)]


def accumulate_world_levels(
    local: np.ndarray, parents: np.ndarray, levels: list[np.ndarray]
) -> np.ndarray:
    """Seviye seviye vektorlestirilmis hiyerarsi zinciri."""
    world = np.empty_like(local)
    for level, indices in enumerate(levels):
        if not len(indices):
            continue
        if level == 0:
            world[indices] = local[indices]
        else:
            world[indices] = world[parents[indices]] @ local[indices]
    return world


class FaceModel:
    """`scripts/04_extract_face_model.py` ciktisinin calisir hali."""

    def __init__(self, data: dict):
        self.region = str(data["region"])
        self.vertex_indices = np.asarray(data["vertexIndices"])
        self.neutral = np.asarray(data["neutral"], dtype=np.float64)
        self.skin_weights = np.asarray(data["skinWeights"], dtype=np.float64)
        self.skin_joints = np.asarray(data["skinJoints"], dtype=np.int64)
        self.blend_deltas = np.asarray(data["blendShapeDeltas"], dtype=np.float64)
        self.blend_channels = np.asarray(data["blendShapeChannels"], dtype=np.int64)
        self.joint_neutral_translation = np.asarray(data["jointNeutralTranslation"], dtype=np.float64)
        self.joint_neutral_rotation = np.asarray(data["jointNeutralRotation"], dtype=np.float64)
        self.joint_parents = np.asarray(data["jointParents"], dtype=np.int64)
        self.gui_control_names = [str(n) for n in data["guiControlNames"]]

        # --- hizli yol: sadece bu bolgenin kullandigi jointleri hesapla
        #
        # Olculdu: deform suresinin %54'u 870 jointlik hiyerarsi zinciriydi
        # ve vertex sayisini dusurmek fayda etmiyordu (sabit maliyet).
        # Goz kapagi bolgesi 870 jointin sadece 128'ini kullaniyor
        # (atalar dahil) ve zincir 8 seviye derinlikte.
        self._active_joints, self._local_parents = self._compute_active_joints()
        self._levels = build_levels(self._local_parents)
        self._joint_remap = np.full(len(self.joint_parents), -1, dtype=np.int64)
        self._joint_remap[self._active_joints] = np.arange(len(self._active_joints))
        self._skin_joints_local = self._joint_remap[self.skin_joints]

        self._bind_inverse = self._compute_bind_inverse()

    def _compute_active_joints(self) -> tuple[np.ndarray, np.ndarray]:
        """Skinlenen jointler + tum atalari. Ata zinciri sart: bir jointin
        dunya matrisi ebeveyninkine bagli."""
        used = set()
        rows, slots = np.nonzero(self.skin_weights > 1e-9)
        used.update(int(j) for j in self.skin_joints[rows, slots])

        needed = set(used)
        for joint in used:
            cursor = joint
            while True:
                parent = int(self.joint_parents[cursor])
                if parent == cursor or parent < 0 or parent in needed:
                    break
                needed.add(parent)
                cursor = parent

        active = np.array(sorted(needed), dtype=np.int64)
        position = {int(j): i for i, j in enumerate(active)}
        local_parents = np.array(
            [position.get(int(self.joint_parents[j]), i) for i, j in enumerate(active)],
            dtype=np.int64,
        )
        return active, local_parents

    @classmethod
    def load(cls, path: Path | str) -> "FaceModel":
        with np.load(Path(path), allow_pickle=False) as data:
            return cls({key: data[key] for key in data.files})

    @property
    def joint_count(self) -> int:
        return len(self.joint_parents)

    @property
    def vertex_count(self) -> int:
        return len(self.neutral)

    def _joint_world(self, joint_outputs: np.ndarray | None) -> np.ndarray:
        """Aktif jointlerin dunya matrisleri. joint_outputs None ise notr poz.

        Donen dizi AKTIF joint sirasinda; global indeksten cevirmek icin
        `_joint_remap` kullanilir.
        """
        active = self._active_joints
        translation = self.joint_neutral_translation[active].copy()
        rotation_degrees = self.joint_neutral_rotation[active].copy()
        scale = np.ones_like(translation)

        if joint_outputs is not None:
            values = np.asarray(joint_outputs, dtype=np.float64)
            expected = self.joint_count * JOINT_STRIDE
            if len(values) != expected:
                raise ValueError(f"joint cikti uzunlugu {len(values)}, beklenen {expected}")
            values = values.reshape(self.joint_count, JOINT_STRIDE)[active]
            # Not: oteleme deltasi addon'da SCALE_FACTOR'e bolunuyor cunku
            # Blender metre kullaniyor. Biz DNA biriminde (cm) calistigimiz
            # icin bolme yok -- notr oteleme de ayni birimde.
            translation += values[:, 0:3]
            rotation_degrees += values[:, 3:6]
            scale += values[:, 6:9]

        rotation = euler_xyz_to_matrix(np.radians(rotation_degrees))
        local = compose(translation, rotation, scale)
        return accumulate_world_levels(local, self._local_parents, self._levels)

    def _compute_bind_inverse(self) -> np.ndarray:
        return np.linalg.inv(self._joint_world(None))

    def deform(
        self,
        joint_outputs: np.ndarray | None = None,
        blend_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Kontrol ciktilarindan vertex konumlari. (vertex_count, 3)"""
        positions = self.neutral.copy()

        # 1) blendshape deltalari (bind uzayinda, skinning'den ONCE)
        if blend_weights is not None and len(self.blend_deltas):
            weights = np.asarray(blend_weights, dtype=np.float64)
            selected = weights[self.blend_channels]
            active = np.nonzero(np.abs(selected) > 1e-9)[0]
            if len(active):
                positions += np.einsum("t,tvc->vc", selected[active], self.blend_deltas[active])

        # 2) linear blend skinning
        world = self._joint_world(joint_outputs)
        skinning = world @ self._bind_inverse  # (aktif joint, 4, 4)

        homogeneous = np.concatenate([positions, np.ones((len(positions), 1))], axis=1)
        result = np.zeros((len(positions), 3), dtype=np.float64)
        for slot in range(self.skin_joints.shape[1]):
            weight = self.skin_weights[:, slot]
            active = np.nonzero(weight > 1e-9)[0]
            if not len(active):
                continue
            matrices = skinning[self._skin_joints_local[active, slot]]
            transformed = np.einsum("vij,vj->vi", matrices, homogeneous[active])
            result[active] += weight[active, None] * transformed[:, :3]

        return result
