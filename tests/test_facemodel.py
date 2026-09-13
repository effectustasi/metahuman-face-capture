"""Ileri model testleri -- cozucunun temeli.

Yanlis vertex hesaplayan bir ileri model, cozucuyu sessizce sacmalatir:
optimizasyon yakinsar ama yanlis yere. Bu yuzden model once kendi
basina dogrulaniyor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy")

from core.facemodel import (  # noqa: E402
    JOINT_STRIDE,
    FaceModel,
    accumulate_world,
    compose,
    euler_xyz_to_matrix,
)

MODEL = ROOT / "mapping" / "_generated" / "face_model_eyelid.npz"
needs_model = pytest.mark.skipif(
    not MODEL.exists(), reason="once scripts/04_extract_face_model.py calistir"
)


@pytest.fixture(scope="module")
def model() -> FaceModel:
    return FaceModel.load(MODEL)


# --------------------------------------------------------------------------
# matris yardimcilari
# --------------------------------------------------------------------------


def test_sifir_euler_birim_matris():
    result = euler_xyz_to_matrix(np.zeros((1, 3)))
    assert np.allclose(result[0], np.eye(3), atol=1e-12)


def test_euler_donme_yonu():
    """X ekseninde 90 derece: Y ekseni Z'ye gitmeli."""
    result = euler_xyz_to_matrix(np.array([[np.pi / 2, 0.0, 0.0]]))[0]
    assert np.allclose(result @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-12)


def test_compose_oteleme_ve_olcek():
    matrix = compose(
        np.array([[1.0, 2.0, 3.0]]),
        np.eye(3)[None],
        np.array([[2.0, 2.0, 2.0]]),
    )[0]
    point = matrix @ np.array([1.0, 0.0, 0.0, 1.0])
    assert np.allclose(point[:3], [3.0, 2.0, 3.0])


def test_hiyerarsi_zincirleniyor():
    """Cocuk, ebeveyninin donusumunu miras almali."""
    local = np.stack([np.eye(4), np.eye(4)])
    local[0, :3, 3] = [10.0, 0.0, 0.0]
    local[1, :3, 3] = [1.0, 0.0, 0.0]
    world = accumulate_world(local, np.array([0, 0]))
    assert np.allclose(world[1][:3, 3], [11.0, 0.0, 0.0])


# --------------------------------------------------------------------------
# ileri model
# --------------------------------------------------------------------------


@needs_model
def test_sifir_kontrolde_notr_poz(model):
    """EN KRITIK TEST. Hicbir kontrol surulmediginde model notr vertexleri
    aynen geri vermeli. Vermiyorsa notr->dunya->bind tersi zinciri
    bozuktur ve tum cozum yanlis yere yakinsar.

    Tolerans float32'ye gore: DNA verisi float32 saklaniyor, koordinatlar
    ~163 cm buyuklugunde, yani beklenen hata ~163 * 1.2e-7 = 2e-5 cm.
    """
    result = model.deform()
    error = np.abs(result - model.neutral).max()
    assert error < 1e-3, f"notr poz {error:.2e} cm sapiyor -- matris zinciri bozuk"


@needs_model
def test_bind_tersi_tam_dogru(model):
    """Zincirin kendisi float64 hassasiyetinde tam olmali; float32 hatasi
    sadece kaynak veriden gelmeli."""
    world = model._joint_world(None)
    identity = world @ model._bind_inverse
    error = np.abs(identity - np.eye(4)).max()
    assert error < 1e-9, f"world @ bind^-1 birim degil: {error:.2e}"


@needs_model
def test_blendshape_vertexleri_oynatiyor(model):
    weights = np.zeros(model.blend_channels.max() + 1)
    weights[model.blend_channels[0]] = 1.0
    result = model.deform(blend_weights=weights)
    assert np.abs(result - model.neutral).max() > 1e-4


@needs_model
def test_blendshape_sifir_agirlikta_etkisiz(model):
    weights = np.zeros(model.blend_channels.max() + 1)
    assert np.allclose(model.deform(blend_weights=weights), model.deform(), atol=1e-12)


@needs_model
def test_joint_deltasi_vertexleri_oynatiyor(model):
    outputs = np.zeros(model.joint_count * JOINT_STRIDE)
    outputs[3] = 5.0  # joint 0, X ekseninde 5 derece
    result = model.deform(joint_outputs=outputs)
    assert np.abs(result - model.neutral).max() > 1e-3


@needs_model
def test_joint_cikti_uzunlugu_dogrulaniyor(model):
    """Yanlis uzunlukta dizi sessizce yanlis okunmasin."""
    with pytest.raises(ValueError):
        model.deform(joint_outputs=np.zeros(10))


@needs_model
def test_deform_deterministik(model):
    outputs = np.zeros(model.joint_count * JOINT_STRIDE)
    outputs[3] = 2.0
    first = model.deform(joint_outputs=outputs)
    second = model.deform(joint_outputs=outputs)
    assert np.array_equal(first, second)


@needs_model
def test_model_meta_verisi(model):
    assert model.region == "eyelid"
    assert model.vertex_count > 0
    assert model.joint_count == len(model.joint_parents)
    assert len(model.gui_control_names) == 174
    assert model.skin_weights.shape[0] == model.vertex_count
