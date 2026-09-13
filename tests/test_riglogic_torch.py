"""Turevlenebilir RigLogic'in GERCEK RigLogic'e karsi dogrulanmasi.

Yeniden yazimin dogrulugu ancak orijinal kutuphaneye karsi kanitlanabilir.
Referans cikti `scripts/06_dump_reference.py` ile Blender icinde uretilip
dosyaya yazildi (torch Blender'da yok, RigLogic torch venv'inde yok).

Testler torch gerektiriyor; yoksa atlanir.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from core.riglogic_torch import TorchRigLogic  # noqa: E402

BEHAVIOR = ROOT / "mapping" / "_generated" / "behavior.npz"
REFERENCE = ROOT / "mapping" / "_generated" / "riglogic_reference.npz"

needs_data = pytest.mark.skipif(
    not (BEHAVIOR.exists() and REFERENCE.exists()),
    reason="once scripts/05_extract_behavior.py ve 06_dump_reference.py calistir",
)


@pytest.fixture(scope="module")
def model():
    return TorchRigLogic.load(BEHAVIOR, dtype=torch.float64)


@pytest.fixture(scope="module")
def reference():
    with np.load(REFERENCE, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


# --------------------------------------------------------------------------
# yapi
# --------------------------------------------------------------------------


@needs_data
def test_kontrol_vektoru_rbf_dahil(model):
    """Kontrol vektoru raw + PSD DEGIL; RBF poz kontrolleri de var.

    Ilk yazimda 808 varsaymistim (263 + 545) ve joint grubu girdi
    indeksleri 825'e gidince patladi. Dogrusu 826.
    """
    assert model.control_count == model.raw_count + model.psd_count + model.rbf_count
    assert model.rbf_count > 0, "bu DNA'da 18 RBF poz kontrolu var"


@needs_data
def test_boyutlar(model):
    assert model.gui_count == 174
    assert model.raw_count == 263
    assert model.psd_count == 545
    assert model.joint_output_count == model.joint_count * 9


# --------------------------------------------------------------------------
# gercek RigLogic'e karsi
# --------------------------------------------------------------------------


@needs_data
def test_gercek_riglogic_ile_ayni(model, reference):
    """EN KRITIK TEST. Ayni GUI girdisi -> ayni joint/blendshape ciktisi.

    Tolerans float32'ye gore: DNA katsayilari float32 saklaniyor, joint
    ciktilarinin olcegi ~166, yani beklenen bagil hata ~1e-7.
    """
    gui = torch.tensor(reference["gui"], dtype=torch.float64)
    with torch.no_grad():
        joints, blends = model(gui)

    joint_reference = torch.tensor(reference["jointOutputs"], dtype=torch.float64)
    blend_reference = torch.tensor(reference["blendOutputs"], dtype=torch.float64)

    joint_scale = joint_reference.abs().max()
    joint_error = (joints - joint_reference).abs().max() / joint_scale
    blend_error = (blends - blend_reference).abs().max()

    assert joint_error < 1e-5, f"joint bagil hata {joint_error:.2e}"
    assert blend_error < 1e-5, f"blendshape hata {blend_error:.2e}"


@needs_data
def test_notr_girdi_tam_sifir(model, reference):
    """Sifir GUI -> sifir cikti, yuvarlama hatasi bile olmamali."""
    gui = torch.zeros(1, model.gui_count, dtype=torch.float64)
    with torch.no_grad():
        joints, blends = model(gui)
    assert joints.abs().max() == 0.0
    assert blends.abs().max() == 0.0


@needs_data
def test_psd_degerleri_carpan_degil(model, reference):
    """PSD ciktisi refere edilen kontrollerin DUZ carpimi.

    Once her terimi kendi `psdValues` degeriyle carpmistim; "hepsi +1"
    orneginde 3.0 hata verdi (PSD 453: deger 4.0 x 1.0 -> benim 4.0,
    RigLogic 1.0). Bu test o regresyonu tutar.
    """
    gui = torch.ones(1, model.gui_count, dtype=torch.float64)
    with torch.no_grad():
        _, blends = model(gui)
    reference_blends = torch.tensor(reference["blendOutputs"][1:2], dtype=torch.float64)
    assert (blends - reference_blends).abs().max() < 1e-5


@needs_data
def test_yogun_ve_gruplu_ayni_sonuc(reference):
    """121 grubu tek matriste birlestirmek sonucu degistirmemeli."""
    gui = torch.tensor(reference["gui"], dtype=torch.float64)
    grouped = TorchRigLogic.load(BEHAVIOR, dtype=torch.float64, dense_joints=False)
    dense = TorchRigLogic.load(BEHAVIOR, dtype=torch.float64, dense_joints=True)
    with torch.no_grad():
        a, _ = grouped(gui)
        b, _ = dense(gui)
    assert (a - b).abs().max() < 1e-6


# --------------------------------------------------------------------------
# turevlenebilirlik -- butun mesele bu
# --------------------------------------------------------------------------


@needs_data
def test_gradyan_akiyor(model):
    gui = torch.zeros(1, model.gui_count, dtype=torch.float64, requires_grad=True)
    joints, blends = model(gui)
    (joints.sum() + blends.sum()).backward()
    assert gui.grad is not None
    assert torch.isfinite(gui.grad).all()


@needs_data
def test_gradyan_numerik_ile_ortusuyor(model):
    """Autograd'in verdigi gradyan sonlu farkla dogrulanmali -- zincirde
    yanlis bir tureve varsa burada yakalanir."""
    torch.manual_seed(0)
    base = torch.rand(1, model.gui_count, dtype=torch.float64) * 0.5

    def cost(x):
        joints, blends = model(x)
        return (joints**2).sum() + (blends**2).sum()

    x = base.clone().requires_grad_(True)
    cost(x).backward()
    analytic = x.grad[0]

    # birkac kontrolu ornekle; hepsini test etmek yavas
    eps = 1e-6
    for index in (0, 17, 60, 120, 173):
        plus, minus = base.clone(), base.clone()
        plus[0, index] += eps
        minus[0, index] -= eps
        with torch.no_grad():
            numeric = (cost(plus) - cost(minus)) / (2 * eps)
        assert abs(numeric - analytic[index]) < 1e-3 * max(1.0, abs(numeric)), (
            f"kontrol {index}: autograd {analytic[index]:.6f} vs numerik {numeric:.6f}"
        )


@needs_data
def test_batch_bagimsiz(model):
    """Batch'teki kareler birbirini etkilememeli."""
    torch.manual_seed(1)
    gui = torch.rand(4, model.gui_count, dtype=torch.float64) * 0.4
    with torch.no_grad():
        batched, _ = model(gui)
        singles = torch.cat([model(gui[i : i + 1])[0] for i in range(4)])
    assert (batched - singles).abs().max() < 1e-12
