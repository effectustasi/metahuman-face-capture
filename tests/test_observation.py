"""Gozlem uzayi testleri.

Asil soru: BASKA bir kafadan gelen landmark'lar karakterin uzayina
tasindiginda kimlik farki gercekten iptal oluyor mu. Olmuyorsa cozucu
ifade kontrollerini kimlik icin harcar ve butun zincir bosa gider.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy")

from core.observation import ObservationSpace, prepare, procrustes  # noqa: E402


def rotation_matrix(yaw: float, pitch: float) -> np.ndarray:
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    return np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array(
        [[1, 0, 0], [0, cp, -sp], [0, sp, cp]]
    )


@pytest.fixture
def scene():
    """Karakter notru, 'baska bir insanin' notru ve bir ifade."""
    rng = np.random.default_rng(7)
    count = 60
    model_neutral = rng.normal(0, 4.0, (count, 3))          # DNA cm
    # kimlik farki: farkli oranlar + genel bir kayma. Ifadeyle alakasiz.
    identity = model_neutral * np.array([1.18, 0.86, 1.05]) + rng.normal(0, 0.6, (count, 3))
    expression = rng.normal(0, 0.35, (count, 3))            # gercek hareket
    return model_neutral, identity, expression, count


def as_frame(points: np.ndarray, resolution=(640, 480)) -> tuple[list, tuple]:
    """Izotropik noktalari MediaPipe'in normalize kordinatina geri cevirir."""
    width, height = resolution
    raw = points.copy()
    raw[:, 1] *= width / height
    return raw.tolist(), resolution


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------


def test_prepare_en_boy_oranini_duzeltiyor():
    """MediaPipe y'yi YUKSEKLIGE boluyor; kare olmayan goruntude
    duzeltilmezse yuz dikey ezik okunur."""
    points = [[0.5, 0.5, 0.0], [0.5, 1.0, 0.0]]
    out = prepare(points, (640, 480))
    assert out[1][1] - out[0][1] == pytest.approx(0.5 * 480 / 640)


def test_prepare_kare_goruntude_degistirmiyor():
    points = [[0.2, 0.7, -0.1]]
    assert prepare(points, (1024, 1024)) == pytest.approx(np.array(points))


def test_prepare_alt_kume_seciyor():
    points = np.arange(30, dtype=float).reshape(10, 3)
    out = prepare(points, (100, 100), landmark_ids=[0, 4, 9])
    assert out.shape == (3, 3)
    assert out[1] == pytest.approx(points[4])


def test_prepare_bozuk_girdiyi_reddediyor():
    with pytest.raises(ValueError):
        prepare([[0.0, 0.0]], (100, 100))
    with pytest.raises(ValueError):
        prepare([[0.0, 0.0, 0.0]], (0, 100))


# --------------------------------------------------------------------------
# procrustes
# --------------------------------------------------------------------------


def test_procrustes_donusumu_geri_buluyor():
    rng = np.random.default_rng(1)
    source = rng.normal(0, 1, (40, 3))
    rotation = rotation_matrix(0.4, -0.2)
    target = 2.5 * (rotation @ source.T).T + np.array([3.0, -1.0, 0.5])
    scale, r, t = procrustes(source, target)
    assert scale == pytest.approx(2.5, abs=1e-9)
    assert np.abs(scale * (r @ source.T).T + t - target).max() < 1e-9


def test_procrustes_yansima_uretmiyor():
    """Aynalanmis veri geldiginde det=-1 rotasyon dondurmek yuzu ters
    cevirir; SVD duzeltmesi bunu engellemeli."""
    rng = np.random.default_rng(2)
    source = rng.normal(0, 1, (40, 3))
    target = source * np.array([1.0, 1.0, -1.0])
    _, rotation, _ = procrustes(source, target)
    assert np.linalg.det(rotation) > 0


# --------------------------------------------------------------------------
# kimlik iptali -- asil test
# --------------------------------------------------------------------------


def bolgesel_ifade(count, rng, olcek=0.35):
    """Gercek ifadeler BOLGESELDIR: agiz oynarken alin durur. Kuresel
    rastgele hareket poz ile ayirt edilemez, o yuzden testler bolgesel."""
    expression = np.zeros((count, 3))
    expression[: count // 3] = rng.normal(0, olcek, (count // 3, 3))
    return expression


def test_kimlik_farki_iptal_oluyor(scene):
    """Farkli bir yuzun ifadesi karakterin uzayinda dogru yer degistirme
    olarak cikmali. Bu testin varlik sebebi budur.

    Rijit altkume ile olculdu -- ayni veride tam yuz Procrustes %11.3
    siziyor, altkumeyle sifira iniyor. Altkume susleme degil, sart."""
    model_neutral, identity, _, count = scene
    rng = np.random.default_rng(21)
    expression = bolgesel_ifade(count, rng)

    space = ObservationSpace(model_neutral, range(count))
    space.learn_neutral([as_frame(identity)])
    hareketli = [
        as_frame(identity + bolgesel_ifade(count, rng, 0.4)) for _ in range(12)
    ]
    space.learn_rigid_subset(hareketli)

    moved = space.to_character(*as_frame(identity + expression))
    beklenen = model_neutral + space.scale * (space.rotation @ expression.T).T
    assert np.abs(moved - beklenen).max() < 1e-6


def test_rijit_altkume_sizintiyi_dusuruyor(scene):
    """Altkumesiz ne kadar kotu oldugunu sabitler -- birisi altkumeyi
    kaldirmak isterse bedelini burada gorur."""
    model_neutral, identity, _, count = scene
    rng = np.random.default_rng(21)
    expression = bolgesel_ifade(count, rng)

    def sizinti(mask_frames):
        space = ObservationSpace(model_neutral, range(count))
        space.learn_neutral([as_frame(identity)])
        if mask_frames:
            space.learn_rigid_subset(mask_frames)
        moved = space.to_character(*as_frame(identity + expression))
        beklenen = model_neutral + space.scale * (space.rotation @ expression.T).T
        return np.abs(moved - beklenen).max()

    hareketli = [as_frame(identity + bolgesel_ifade(count, rng, 0.4)) for _ in range(12)]
    assert sizinti(None) > 0.05          # tam yuz: olculen %11.3
    assert sizinti(hareketli) < 1e-6     # altkume: olculen %0


def test_kuresel_hareket_pozdan_ayrilamiyor(scene):
    """Bilinen SINIR: her landmark bagimsiz oynarsa hareketin bir kismi
    rijit donusum gibi gorunur ve ayiklanir. Olculdu: %25 kayip.
    Gercek yuzlerde olmaz cunku ifadeler bolgeseldir; yine de cozucu
    'her sey oynuyor' bir cozume kaymasin diye burada yaziliyor."""
    model_neutral, identity, expression, count = scene
    space = ObservationSpace(model_neutral, range(count))
    space.learn_neutral([as_frame(identity)])
    moved = space.to_character(*as_frame(identity + expression))
    beklenen = model_neutral + space.scale * (space.rotation @ expression.T).T
    assert np.abs(moved - beklenen).max() > 0.1


def test_notr_gozlem_notr_hedef_veriyor(scene):
    """Ifadesiz kare karakterin notrunu birebir vermeli -- vermezse
    cozucu yoktan ifade uretir."""
    model_neutral, identity, _, _ = scene
    space = ObservationSpace(model_neutral, range(len(model_neutral)))
    space.learn_neutral([as_frame(identity)])
    assert np.abs(space.to_character(*as_frame(identity)) - model_neutral).max() < 1e-9


def test_kafa_pozu_ifade_sanilmiyor(scene):
    """Kafa donunce/yaklasinca hedef degismemeli."""
    model_neutral, identity, _, _ = scene
    space = ObservationSpace(model_neutral, range(len(model_neutral)))
    space.learn_neutral([as_frame(identity)])

    center = identity.mean(0)
    posed = 1.3 * (rotation_matrix(0.35, 0.15) @ (identity - center).T).T + center + [0.4, -0.2, 0.1]
    assert np.abs(space.to_character(*as_frame(posed)) - model_neutral).max() < 1e-6


def test_notr_birden_cok_kareyi_ortaliyor(scene):
    """Tek kare MediaPipe titremesini de notr sanar."""
    model_neutral, identity, _, count = scene
    rng = np.random.default_rng(11)
    frames = [as_frame(identity + rng.normal(0, 0.02, (count, 3))) for _ in range(30)]
    space = ObservationSpace(model_neutral, range(count))
    assert space.learn_neutral(frames) == 30
    assert np.abs(space.neutral - identity).max() < 0.03


# --------------------------------------------------------------------------
# rijit altkume
# --------------------------------------------------------------------------


def test_rijit_altkume_oynak_landmarklari_diskiyor(scene):
    """Agiz acilinca cene hareketini kafa donusu sanmamak icin, poz
    ayiklamasi sadece kipirdamayan landmark'lara bakmali."""
    model_neutral, identity, _, count = scene
    rng = np.random.default_rng(3)
    oynak = np.arange(count // 3)            # 'agiz' bolgesi
    frames = []
    for _ in range(12):
        points = identity.copy()
        points[oynak] += rng.normal(0, 1.2, (len(oynak), 3))
        frames.append(as_frame(points))

    space = ObservationSpace(model_neutral, range(count))
    space.learn_neutral([as_frame(identity)])
    mask = space.learn_rigid_subset(frames)
    assert len(set(mask) & set(oynak.tolist())) == 0, "oynak landmark rijit sayildi"


def test_notrsuz_cagri_hata_veriyor(scene):
    model_neutral, identity, _, _ = scene
    space = ObservationSpace(model_neutral, range(len(model_neutral)))
    with pytest.raises(RuntimeError):
        space.to_character(*as_frame(identity))


def test_uyumsuz_notr_boyutu_reddediliyor():
    with pytest.raises(ValueError):
        ObservationSpace(np.zeros((10, 3)), range(9))
