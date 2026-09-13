"""Cekimi cozer ve ARKit eslemesiyle YAN YANA karsilastirir.

Sorunun tamami su: cozucu, ARKit blendshape eslemesinden daha iyi mi.
Tek basina "cozucu 1.4 mm hata verdi" bir sey soylemiyor; ayni karede
ARKit yolu ne veriyor, onun yaninda anlam kazaniyor.

Olcut: her iki yolun urettigi GUI kontrolleriyle karakterin landmark'lari
hesaplanir, gozlemle karsilastirilir. Gozlem cozucunun hedefi oldugu icin
bu olcut ona yanli -- ama ARKit yolunun ayni gozlemi ne kadar
aciklayabildigi yine de gercek bir bilgi.

    python scripts/10_solve_take.py --take takes/x.jsonl [--frames 60,120]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.mapping import Mapping  # noqa: E402
from core.observation import ObservationSpace  # noqa: E402
from core.solver import LandmarkSolver  # noqa: E402
from core.take import read_take  # noqa: E402

GENERATED = ROOT / "mapping" / "_generated"


def gui_from_arkit(mapping: Mapping, blendshapes: dict, names: list[str]) -> torch.Tensor:
    """ARKit skorlari -> (174,) GUI vektoru.

    Esleme 'CTRL_C_jaw.y' yaziyor, DNA 'CTRL_C_jaw.ty'; eksen adi
    ceviriliyor. Karsiligi olmayan eksen sessizce atlanmiyor, sayiliyor."""
    vector = torch.zeros(len(names))
    index = {name: i for i, name in enumerate(names)}
    missing = 0
    for axis_key, value in mapping.apply(blendshapes).items():
        bone, axis = axis_key.rsplit(".", 1)
        position = index.get(f"{bone}.t{axis}")
        if position is None:
            missing += 1
            continue
        vector[position] = value
    return vector, missing


def control_influence(solver, batch: int = 16) -> "torch.Tensor":
    """Her GUI kontrolunun landmark'lar uzerindeki etkisi (mm, deger=1.0'da).

    Neden gerekli: kontroller esit agirlikta degil. `CTRL_R_ear_up` sadece
    7 landmark'i 0.047 mm oynatiyor -- cozucu onu 1.00'a itse bile hedefe
    etkisi yok, ama DEGERE gore siralanan bir listede en ustte gorunup
    "cozucu kulagi acmis" gibi okunuyor. Olculdu ve yanlis alarm verdi.
    """
    with torch.no_grad():
        neutral = solver.landmarks_from_gui(
            torch.zeros(1, solver.rig.gui_count, dtype=solver.dtype, device=solver.device)
        )[0]
        out = torch.zeros(solver.rig.gui_count)
        for start in range(0, solver.rig.gui_count, batch):
            count = min(batch, solver.rig.gui_count - start)
            probe = torch.zeros(count, solver.rig.gui_count, dtype=solver.dtype, device=solver.device)
            for row in range(count):
                probe[row, start + row] = 1.0
            moved = (solver.landmarks_from_gui(probe) - neutral).norm(dim=2).mean(dim=1) * 10
            out[start : start + count] = moved.cpu()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--take", required=True, type=Path)
    parser.add_argument("--neutral", type=int, default=30, help="notr icin ilk N kare")
    parser.add_argument("--frames", help="cozulecek kare numaralari, virgullu. Bos = 5 esit aralikli")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--out", type=Path, help="cozulen GUI vektorlerini JSON yaz")
    args = parser.parse_args()

    frames = [f for f in read_take(args.take) if f.lm]
    if not frames:
        raise SystemExit(f"{args.take} icinde landmark'li kare yok -- detector'i --landmarks ile calistir")
    print(f"{len(frames)} landmark'li kare")

    solver = LandmarkSolver(
        GENERATED / "behavior.npz",
        GENERATED / "face_model_landmarks.npz",
        GENERATED / "correspondence" / "correspondence.json",
    )
    names = solver.rig.gui_control_names
    with torch.no_grad():
        model_neutral = solver.landmarks_from_gui(torch.zeros(1, solver.rig.gui_count))[0].numpy()

    space = ObservationSpace(model_neutral, solver.landmark_ids)
    pairs = [(f.lm, f.res or [1, 1]) for f in frames]
    used = space.learn_neutral(pairs[: args.neutral])
    rigid = space.learn_rigid_subset(pairs)
    print(f"notr {used} kareden, rijit altkume {len(rigid)}/{solver.landmark_count} landmark")
    print(f"kimlik olcegi {space.scale:.4f} (gozlem birimi -> DNA cm)")

    if args.frames:
        chosen = [int(x) for x in args.frames.split(",")]
        selected = [f for f in frames if f.frame in chosen]
    else:
        step = max(1, len(frames) // 5)
        selected = frames[::step][:5]

    mapping = Mapping.load()
    influence = control_influence(solver)
    results = []
    print(f"\n{'kare':>6} {'cozucu mm':>11} {'ARKit mm':>11} {'kontrol':>8}   en guclu kontroller")
    print("-" * 96)
    for frame in selected:
        observed = torch.tensor(space.to_character(frame.lm, frame.res), dtype=torch.float32)
        solved, info = solver.solve(observed, iterations=args.iterations)

        arkit, missing = gui_from_arkit(mapping, frame.bs, names)
        with torch.no_grad():
            from core.solver import rigid_align

            predicted = solver.landmarks_from_gui(arkit[None])[0]
            target = rigid_align(observed, predicted)
            arkit_rms = float(((predicted - target) ** 2).sum(1).mean().sqrt()) * 10

        # ETKIYE gore sirala, degere gore degil (bkz. control_influence)
        top = sorted(
            (
                (abs(float(v)) * float(influence[i]), float(v), names[i])
                for i, v in enumerate(solved)
                if abs(float(v)) > 0.05
            ),
            reverse=True,
        )[:4]
        label = ", ".join(f"{n.replace('CTRL_','')}={v:.2f}" for _, v, n in top) or "-"
        print(
            f"{frame.frame:>6} {info['rmsMillimeters']:>11.3f} {arkit_rms:>11.3f} "
            f"{info['activeControls']:>8}   {label}"
        )
        results.append(
            {
                "frame": frame.frame,
                "t": frame.t,
                "solverRms": info["rmsMillimeters"],
                "arkitRms": arkit_rms,
                "activeControls": info["activeControls"],
                "gui": {names[i]: round(float(v), 4) for i, v in enumerate(solved) if abs(float(v)) > 0.01},
            }
        )
        if missing:
            print(f"       uyari: eslemedeki {missing} eksen DNA'da yok")

    if args.out:
        args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
