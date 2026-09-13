"""UDP'ye ne geldigini gosterir. Blender acmadan agi dogrulamak icin.

    python scripts/udp_monitor.py --port 11111

Katman ayirmak icin: monitor paket goruyor ama Blender'da yuz oynamiyorsa
sorun ag tarafinda degil, Blender tarafindadir.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.mapping import Mapping  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=11111)
    parser.add_argument("--count", type=int, default=0, help="0 = sinirsiz")
    parser.add_argument("--top", type=int, default=4, help="kac kanal gosterilsin")
    args = parser.parse_args()

    try:
        mapping = Mapping.load()
    except OSError:
        mapping = None

    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listener.bind(("0.0.0.0", args.port))
    listener.settimeout(5.0)

    print(f"udp://0.0.0.0:{args.port} dinleniyor. Ctrl+C ile cik.")

    received = 0
    unknown_reported = False
    last_report = time.perf_counter()
    since_report = 0

    try:
        while True:
            try:
                payload, sender = listener.recvfrom(65535)
            except socket.timeout:
                print("  ... 5 saniyedir paket yok")
                continue

            try:
                data = json.loads(payload.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                print(f"  ! cozulemeyen paket, {len(payload)} bayt")
                continue

            received += 1
            since_report += 1
            blendshapes = data.get("bs") or {}

            if mapping and not unknown_reported:
                unknown = mapping.unknown_shapes(blendshapes)
                if unknown:
                    print(f"  ! tabloda olmayan kanallar: {unknown}")
                    unknown_reported = True

            now = time.perf_counter()
            if now - last_report >= 1.0:
                top = sorted(blendshapes.items(), key=lambda kv: -abs(kv[1]))[: args.top]
                summary = "  ".join(f"{name}={value:.2f}" for name, value in top)
                print(
                    f"kare {data.get('frame'):>5}  t={data.get('t', 0):6.2f}  "
                    f"{since_report:3d} pkt/sn  {summary}"
                )
                last_report = now
                since_report = 0

            if args.count and received >= args.count:
                break
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\ntoplam {received} paket, kaynak {sender[0] if received else '-'}")


if __name__ == "__main__":
    main()
