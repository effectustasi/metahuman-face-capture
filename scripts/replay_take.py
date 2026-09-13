"""Bir .jsonl cekimini gercek zamanli UDP'ye basar -- sahte detector.

Canli yolu MediaPipe olmadan test etmek icin. Blender'daki charface.live
operator'u acisindan bu, webcam'den gelen veriden ayirt edilemez; ayni
paket semasi, ayni tempo.

    python scripts/replay_take.py tests/fixtures/take_5s.jsonl --loop

Boylece canli zincirin Blender yarisi (UDP alimi, esleme, RigLogic
guncellemesi, viewport tazeleme) detector kurulmadan dogrulanabilir.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.take import read_take  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("take", type=Path)
    parser.add_argument("--udp", default="127.0.0.1:11111")
    parser.add_argument("--loop", action="store_true", help="bitince bastan basla")
    parser.add_argument("--speed", type=float, default=1.0, help="1.0 = gercek zaman")
    args = parser.parse_args()

    host, _, port = args.udp.partition(":")
    address = (host, int(port))
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    frames = list(read_take(args.take))
    if not frames:
        raise SystemExit("cekim bos")

    duration = frames[-1].t - frames[0].t
    print(f"{len(frames)} kare, {duration:.1f} sn -> udp://{host}:{port}")
    print("durdurmak icin Ctrl+C")

    cycle = 0
    try:
        while True:
            wall_start = time.perf_counter()
            take_start = frames[0].t

            for frame in frames:
                # cekimin kendi zaman damgasina gore bekle
                target = (frame.t - take_start) / args.speed
                drift = target - (time.perf_counter() - wall_start)
                if drift > 0:
                    time.sleep(drift)

                # zaman damgasini surekli ileri tasi, yoksa dongude
                # one-euro filtresi zamanin geri gittigini gorup sifirlaniyor
                payload = frame.to_json()
                payload["t"] = round(cycle * (duration + 1 / 30) + frame.t, 6)
                sender.sendto(json.dumps(payload, separators=(",", ":")).encode("utf-8"), address)

            cycle += 1
            print(f"  tur {cycle} bitti")
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\ndurduruldu")


if __name__ == "__main__":
    main()
