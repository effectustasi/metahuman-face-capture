"""Live Link Face paketini gercek telefonla dogrular.

Paket duzeni Epic tarafindan resmi belgelenmemis; cozucu belgelenmemis
bir duzene gore yazildi ve sentetik paketle test edildi. GERCEK telefonla
dogrulanmasi sart -- bu script onun icin.

    python scripts/llf_probe.py

Blender'daki canli yakalama ayni portu tutuyorsa once ESC'ye bas;
bir portu ayni anda tek surec dinleyebilir.

Telefon tarafi:
  1. App Store > "Live Link Face" (Epic Games, ucretsiz)
  2. Sol ust dislii > Live Link > Add Target > PC'nin IP'si, port 11111
  3. Telefon ve PC ayni WiFi'da olmali
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.livelink import CHANNELS, DecodeError, decode  # noqa: E402


def hex_dump(data: bytes, limit: int = 96) -> str:
    chunk = data[:limit]
    lines = []
    for offset in range(0, len(chunk), 16):
        row = chunk[offset : offset + 16]
        hexa = " ".join(f"{b:02x}" for b in row)
        text = "".join(chr(b) if 32 <= b < 127 else "." for b in row)
        lines.append(f"  {offset:04x}  {hexa:<47}  {text}")
    if len(data) > limit:
        lines.append(f"  ... toplam {len(data)} bayt")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=11111)
    parser.add_argument("--count", type=int, default=5, help="kac paket incelensin")
    args = parser.parse_args()

    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        listener.bind(("0.0.0.0", args.port))
    except OSError as error:
        raise SystemExit(
            f"port {args.port} acilamadi: {error}\n"
            "Blender'da canli yakalama aciksa once ESC'ye bas."
        )
    listener.settimeout(20.0)

    print(f"udp://0.0.0.0:{args.port} dinleniyor. Telefonda yayini baslat.")
    print("(20 saniye icinde paket gelmezse cikar)\n")

    for index in range(args.count):
        try:
            data, sender = listener.recvfrom(65535)
        except socket.timeout:
            raise SystemExit(
                "Paket gelmedi. Kontrol et:\n"
                "  - telefon ve PC ayni WiFi'da mi\n"
                "  - Live Link Face'te hedef IP dogru mu\n"
                "  - Windows guvenlik duvari UDP 11111'i engelliyor olabilir"
            )

        print(f"--- paket {index + 1}  ({len(data)} bayt, {sender[0]}) ---")
        if index == 0:
            print(hex_dump(data))

        try:
            channels, meta = decode(data)
        except DecodeError as error:
            print(f"  COZULEMEDI: {error}")
            print("  Yukaridaki hex dokumunu paylas, cozucuyu duzeltelim.\n")
            continue

        print(f"  cozum yolu : {meta['method']}", end="")
        if meta["method"] == "scan":
            print("   <-- YAPISAL COZUM TUTMADI, duzen farkli olabilir")
        else:
            print(f"   cihaz={meta.get('device')} konu={meta.get('subject')} fps={meta.get('fps')}")

        aktif = sorted(
            ((n, v) for n, v in channels.items() if n in CHANNELS[:52] and v > 0.05),
            key=lambda kv: -kv[1],
        )[:6]
        print("  en aktif   :", ", ".join(f"{n}={v:.2f}" for n, v in aktif) or "hepsi ~0")
        print(
            "  kafa       : "
            f"yaw={channels.get('HeadYaw', 0):+.3f} "
            f"pitch={channels.get('HeadPitch', 0):+.3f} "
            f"roll={channels.get('HeadRoll', 0):+.3f}  (radyan)"
        )
        print(f"  tongueOut  : {channels.get('TongueOut', 0):.3f}\n")

    print("Dogrulama listesi:")
    print("  [ ] agzini ac    -> JawOpen buyuyor mu")
    print("  [ ] dudak kapat  -> MouthClose buyuyor mu")
    print("  [ ] dil cikar    -> tongueOut buyuyor mu")
    print("  [ ] kafani cevir -> HeadYaw isaret degistiriyor mu")
    print("Hepsi tutuyorsa cozucu dogru; Blender'da canli yakalamayi baslatabilirsin.")


if __name__ == "__main__":
    main()
