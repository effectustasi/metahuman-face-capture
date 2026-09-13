"""Cozucu testleri: sentetik gidis-donus.

Cozucunun dogrulugu en net su testle olculur: bilinen bir ifadeden
landmark uret, cozucuye ver, ayni kontrolleri geri bulabiliyor mu.
Ileri model dogru ama cozucu bozuksa burada yakalanir.

torch ve uretilmis veri gerektirir; yoksa atlanir.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

GENERATED = ROOT / "mapping" / "_generated"
BEHAVIOR = GENERATED / "behavior.npz"
FACE_MODEL = GENERATED / "face_model_landmarks.npz"
CORRESPONDENCE = GENERATED / "correspondence" / "correspondence.json"

needs_data = pytest.mark.skipif(
    not (BEHAVIOR.exists() and FACE_MODEL.exists() and CORRESPONDENCE.exists()),
    reason="once scripts/04,05 + karsilik uretimi calistir",
)

if BEHAVIOR.exists():
    from core.solver import LandmarkSolver, TorchFaceModel, rigid_align  # noqa: E402


@pytest.fixture(scope="module")
def solver():
    return LandmarkSolver(BEHAVIOR, FACE_MODEL, CORRESPONDENCE)


@pytest.fixture(scope="module")
def expression(solver):
    """Gercekci bir ifade: birkac kontrol acik, gerisi kapali."""
    names = solver.rig.gui_control_names
    active = {
        "CTRL_C_jaw.ty": 0.6,
        "CTRL_L_mouth_cornerPull.ty": 0.8,
        "CTRL_R_mouth_cornerPull.ty": 0.8,
        "CTRL_L_eye_blink.ty": 0.5,
        "CTRL_R_eye_blink.ty": 0.5,
        "CTRL_L_brow_raiseIn.ty": 0.7,
    }
    gui = torch.zeros(1, solver.rig.gui_count)
    for name, value in active.items():
        gui[0, names.index(name)] = value
    return gui, active


# --------------------------------------------------------------------------
# turevlenebilir deform, numpy ikiziyle ayni mi
# --------------------------------------------------------------------------


@needs_data
def test_torch_deform_numpy_ile_ayni():
    """Turevlenebilir surum, dogrulanmis numpy surumuyle ayni sonucu
    vermeli. Sapma varsa cozucu yanlis geometriye yakinsar."""
    from core.facemodel import FaceModel

    reference = FaceModel.load(FACE_MODEL)
    model = TorchFaceModel.load(FACE_MODEL, dtype=torch.float64)

    joints = np.zeros(reference.joint_count * 9)
    joints[3] = 3.0
    joints[12] = -2.0
    weights = np.zeros(reference.blend_channels.max() + 1)
    weights[reference.blend_channels[0]] = 0.7

    expected = reference.deform(joint_outputs=joints, blend_weights=weights)
    actual = model(
        torch.tensor(joints, dtype=torch.float64)[None],
        torch.tensor(weights, dtype=torch.float64)[None],
    )[0].numpy()
    assert np.abs(expected - actual).max() < 1e-9


@needs_data
def test_sifir_kontrolde_notr(solver):
    gui = torch.zeros(1, solver.rig.gui_count)
    with torch.no_grad():
        vertices = solver.model(*solver.rig(gui))[0]
    assert np.abs(vertices.numpy() - solver.model.neutral.numpy()).max() < 1e-3


# --------------------------------------------------------------------------
# rijit hizalama
# --------------------------------------------------------------------------


@needs_data
def test_rijit_hizalama_pozu_ayikliyor(solver, expression):
    """Kafa donmus/kaymis olsa bile hizalama sonrasi ayni yere oturmali --
    cozucunun kafa pozunu ifade sanmamasi buna bagli."""
    gui, _ = expression
    with torch.no_grad():
        points = solver.landmarks_from_gui(gui)[0]

    angle = torch.tensor(0.3)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moved = (rotation @ points.T).T * 1.4 + torch.tensor([0.5, -0.2, 0.3])

    aligned = rigid_align(moved, points)
    assert (aligned - points).abs().max() < 1e-3


# --------------------------------------------------------------------------
# sentetik gidis-donus -- asil test
# --------------------------------------------------------------------------


@needs_data
def test_bilinen_ifade_geri_bulunuyor(solver, expression):
    """Cozucunun varlik sebebi bu test."""
    gui, active = expression
    names = solver.rig.gui_control_names
    with torch.no_grad():
        observed = solver.landmarks_from_gui(gui)[0]

    result, info = solver.solve(observed, align=False)

    errors = {name: abs(float(result[names.index(name)]) - value) for name, value in active.items()}
    worst = max(errors.values())
    assert worst < 0.05, f"en kotu kontrol hatasi {worst:.3f}: {errors}"
    assert info["rmsMillimeters"] < 0.05, f"RMS {info['rmsMillimeters']:.4f} mm"


@needs_data
def test_seyreklik_yanlis_tetiklemeyi_onluyor(solver, expression):
    """L1 olmadan 23 kontrol yanlislikla tetikleniyordu; problem kotu
    kosullu ve farkli kombinasyonlar benzer landmark dizilimi uretiyor."""
    gui, active = expression
    with torch.no_grad():
        observed = solver.landmarks_from_gui(gui)[0]

    result, info = solver.solve(observed, align=False)
    assert info["activeControls"] <= len(active) + 2, (
        f"{info['activeControls']} kontrol aktif, beklenen ~{len(active)}"
    )


@needs_data
def test_yanlilik_giderme_iyilestiriyor(solver, expression):
    """L1 tahminleri sifira dogru buzer; ikinci asama bunu duzeltmeli."""
    gui, active = expression
    names = solver.rig.gui_control_names
    with torch.no_grad():
        observed = solver.landmarks_from_gui(gui)[0]

    ham, _ = solver.solve(observed, align=False, debias=False)
    duzeltilmis, _ = solver.solve(observed, align=False, debias=True)

    def mean_error(vector):
        return sum(abs(float(vector[names.index(n)]) - v) for n, v in active.items()) / len(active)

    assert mean_error(duzeltilmis) < mean_error(ham)


@needs_data
def test_notr_gozlemde_notr_cozum(solver):
    """Ifadesiz yuz -> tum kontroller ~0. Aksi halde cozucu yoktan
    ifade uretiyor demektir."""
    gui = torch.zeros(1, solver.rig.gui_count)
    with torch.no_grad():
        observed = solver.landmarks_from_gui(gui)[0]

    result, info = solver.solve(observed, align=False, iterations=120)
    assert result.abs().max() < 0.1, f"notr gozlemde {info['activeControls']} kontrol tetiklendi"


# --------------------------------------------------------------------------
# karsilik verisi
# --------------------------------------------------------------------------


@needs_data
def test_karsilik_yuklendi(solver):
    assert solver.landmark_count > 400, f"sadece {solver.landmark_count} landmark"


@needs_data
def test_bozuk_agirliklar_eleniyor(solver):
    """Ucgen disina tasan karsiliklar konumu asiri ekstrapole eder;
    olculdu: birkaci -0.87'ye kadar gidiyordu."""
    assert solver.weights.min() >= -0.05 - 1e-6
    assert solver.weights.max() <= 1.05 + 1e-6


@needs_data
def test_agirliklar_birim_toplamli(solver):
    total = solver.weights.sum(dim=1)
    assert (total - 1.0).abs().max() < 1e-3
