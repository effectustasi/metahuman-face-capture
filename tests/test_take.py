"""Take semasi ve sahne karesi planlama testleri."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.take import plan_scene_frames  # noqa: E402




# --------------------------------------------------------------------------
# sahne karesi planlama
# --------------------------------------------------------------------------


def bs(**kwargs):
    return dict(kwargs)


def test_plan_zaman_damgasina_gore_oturuyor():
    """30 fps cekim 24 fps sahneye yazilinca SURE korunmali.

    Kare kareye 1:1 yazim 37 saniyelik cekimi 46 saniye yapiyordu -- %25
    yavas. Kullanicinin fark etmesi zor, cunku animasyon 'calisiyor'."""
    entries = [(i / 30.0, bs(jawOpen=0.5), None) for i in range(300)]  # 10 sn
    plan = plan_scene_frames(entries, fps=24.0, start_frame=1)
    assert max(plan) - min(plan) == 240 - 1, f"10 sn 24 fps'te 240 kare olmali, {max(plan)} bulundu"


def test_plan_ayni_kareye_dusenlerin_EN_YUKSEGINI_aliyor():
    """Goz kirpma 89 ms suruyor; 30->24'te birlesen karelerde sonuncuyu
    almak tepeyi kaybettiriyor."""
    entries = [
        (0.000, bs(eyeBlinkLeft=0.1), None),
        (0.010, bs(eyeBlinkLeft=0.9), None),  # tepe
        (0.020, bs(eyeBlinkLeft=0.2), None),
    ]
    plan = plan_scene_frames(entries, fps=24.0, start_frame=1)
    assert len(plan) == 1, "uc kare ayni sahne karesine dusmeliydi"
    assert plan[1][0]["eyeBlinkLeft"] == 0.9


def test_plan_duraklamayi_koruyor():
    """Cekimin ortasinda duraksama varsa sahne karelerinde de bosluk kalmali."""
    entries = [(0.0, bs(a=1.0), None), (2.0, bs(a=1.0), None)]
    plan = plan_scene_frames(entries, fps=30.0, start_frame=1)
    assert sorted(plan) == [1, 61]


def test_plan_bozuk_zaman_damgasinda_sirali_yaziyor():
    """Hepsi ayni t ise (eski cekimler, replay) sirali yazima dusmeli --
    hepsini tek kareye yigmak yerine."""
    entries = [(0.0, bs(a=float(i)), None) for i in range(5)]
    plan = plan_scene_frames(entries, fps=24.0, start_frame=10)
    assert sorted(plan) == [10, 11, 12, 13, 14]


def test_plan_geri_giden_zaman_damgasini_reddediyor():
    entries = [(0.0, bs(a=1.0), None), (5.0, bs(a=1.0), None), (1.0, bs(a=1.0), None)]
    plan = plan_scene_frames(entries, fps=24.0, start_frame=1)
    assert sorted(plan) == [1, 2, 3]


def test_plan_bos_cekim():
    assert plan_scene_frames([], fps=24.0) == {}


def test_plan_kafa_pozunu_tasiyor():
    head = {"rot": [0.0, 0.0, 0.0, 1.0]}
    plan = plan_scene_frames([(0.0, bs(a=1.0), head)], fps=24.0, start_frame=1)
    assert plan[1][1] is head
