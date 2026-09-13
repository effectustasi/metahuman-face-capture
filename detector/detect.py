"""MediaPipe FaceLandmarker -> ARKit skorlari + kafa pozu.

Bu dosya AYRI bir venv'de kosar (detector/.venv, Python 3.11). Blender ile
tek temasi cikti formatidir: offline .jsonl dosyasi veya canli UDP paketi.
Ikisi de core/take.py'deki semayi kullanir.

Kullanim:

  # video dosyasindan cekim uret
  python detector/detect.py video --input cekim.mp4 --out takes/cekim.jsonl

  # webcam'den canli yayin
  python detector/detect.py live --udp 127.0.0.1:11111

  # webcam'den kaydet (canli yayin + dosya ayni anda)
  python detector/detect.py live --udp 127.0.0.1:11111 --out takes/canli.jsonl

Model dosyasi:
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
  -> detector/models/face_landmarker.task
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.calibration import Profile  # noqa: E402
from core.filters import ChannelFilters  # noqa: E402
from core.take import Frame, write_take  # noqa: E402

DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "face_landmarker.task"


def import_mediapipe():
    try:
        import mediapipe as mp
    except ImportError:
        raise SystemExit(
            "mediapipe bulunamadi. Bu script detector/.venv icinden calistirilmali:\n"
            "  py -3.11 -m venv detector/.venv\n"
            "  detector/.venv/Scripts/python -m pip install -r detector/requirements.txt"
        )
    return mp


def build_landmarker(mp, model_path: Path, running_mode: str, callback=None):
    if not model_path.exists():
        raise SystemExit(
            f"model yok: {model_path}\n"
            "indir: https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task"
        )

    base = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    vision = mp.tasks.vision
    options = vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=getattr(vision.RunningMode, running_mode),
        num_faces=1,
        output_face_blendshapes=True,
        # kafa rotasyonu blendshape'lerden AYRI kanal olarak tasinir;
        # face board'a degil head/neck kemigine uygulanacak
        output_facial_transformation_matrixes=True,
        result_callback=callback,
    )
    return vision.FaceLandmarker.create_from_options(options)


def extract_blendshapes(result) -> dict[str, float]:
    """MediaPipe 52 kategori dondurur, ilki '_neutral'dir ve atilir -> 51 skor."""
    if not result.face_blendshapes:
        return {}
    return {
        category.category_name: round(float(category.score), 5)
        for category in result.face_blendshapes[0]
        if category.category_name != "_neutral"
    }


def extract_landmarks(result) -> list[list[float]] | None:
    """478 ham landmark. Cozucu bunlari kullaniyor; ARKit yolu kullanmiyor.

    x, y goruntude normalize; z **metrik degil** -- MediaPipe onu goreceli
    derinlik olarak veriyor ve olcegi x/y ile ayni degil. Cozucu bu yuzden
    rijit hizalamada olcegi serbest birakiyor."""
    if not result.face_landmarks:
        return None
    return [
        [round(float(p.x), 5), round(float(p.y), 5), round(float(p.z), 5)]
        for p in result.face_landmarks[0]
    ]


def extract_head(result) -> dict | None:
    """4x4 donusum matrisinden quaternion + oteleme."""
    if not result.facial_transformation_matrixes:
        return None

    matrix = result.facial_transformation_matrixes[0]
    rotation = [[float(matrix[row][col]) for col in range(3)] for row in range(3)]
    translation = [float(matrix[row][3]) for row in range(3)]
    return {"rot": matrix_to_quaternion(rotation), "trans": translation}


def matrix_to_quaternion(m) -> list[float]:
    """3x3 rotasyon -> (x, y, z, w). Shepperd yontemi: en buyuk kosegen
    terimden turetip sayisal kararsizligi onler."""
    import math

    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    return [round(v, 6) for v in (x, y, z, w)]


class UdpSender:
    def __init__(self, target: str):
        host, _, port = target.partition(":")
        self.address = (host, int(port))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, frame: Frame) -> None:
        self.socket.sendto(json.dumps(frame.to_json(), separators=(",", ":")).encode("utf-8"), self.address)


def run_video(args):
    mp = import_mediapipe()
    import cv2

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise SystemExit(f"video acilamadi: {args.input}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    landmarker = build_landmarker(mp, args.model, "VIDEO")
    filters = ChannelFilters() if args.filter else None
    profile = Profile.load(args.profile) if args.profile else None

    def frames():
        index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            index += 1
            timestamp_ms = int((index - 1) / fps * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            blendshapes = extract_blendshapes(result)
            if not blendshapes:
                continue  # yuz bulunamayan kare
            t = timestamp_ms / 1000.0
            if profile:
                blendshapes = profile.apply(blendshapes)
            if filters:
                blendshapes = filters(blendshapes, t)

            yield Frame(
                t=t,
                frame=index,
                bs=blendshapes,
                head=extract_head(result),
                lm=extract_landmarks(result) if args.landmarks else None,
                res=[image.shape[1], image.shape[0]] if args.landmarks else None,
            )

    count = write_take(args.out, frames())
    capture.release()
    print(f"{count} kare -> {args.out}")


def run_live(args):
    mp = import_mediapipe()
    import cv2

    sender = UdpSender(args.udp) if args.udp else None
    filters = ChannelFilters() if args.filter else None
    profile = Profile.load(args.profile) if args.profile else None
    recorded: list[Frame] = []
    state = {"index": 0}
    start = time.time()

    def on_result(result, output_image, timestamp_ms):
        blendshapes = extract_blendshapes(result)
        if not blendshapes:
            return
        state["index"] += 1
        t = timestamp_ms / 1000.0
        if profile:
            blendshapes = profile.apply(blendshapes)
        if filters:
            blendshapes = filters(blendshapes, t)

        frame = Frame(
            t=t,
            frame=state["index"],
            bs=blendshapes,
            head=extract_head(result),
            lm=extract_landmarks(result) if args.landmarks else None,
            res=state.get("resolution") if args.landmarks else None,
        )
        if sender:
            sender.send(frame)
        if args.out:
            recorded.append(frame)

    landmarker = build_landmarker(mp, args.model, "LIVE_STREAM", callback=on_result)
    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not capture.isOpened():
        raise SystemExit(f"kamera acilamadi: {args.camera}")

    print("calisiyor. durdurmak icin pencerede q veya konsolda Ctrl+C.")
    last_timestamp_ms = -1
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            # callback goruntuye erisemiyor; cozunurlugu buradan tasiyoruz
            state["resolution"] = [image.shape[1], image.shape[0]]

            # detect_async zaman damgasinin KESIN artmasini sart kosuyor; ayni
            # milisaniyede iki kare gelirse "Input timestamp must be
            # monotonically increasing" atip surec olur. Yuksek fps'te ya da
            # kamera kareleri toplu geldiginde gercekten oluyor.
            timestamp_ms = int((time.time() - start) * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            landmarker.detect_async(mp_image, timestamp_ms)

            if args.preview:
                cv2.imshow("facecap", image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if args.preview:
            cv2.destroyAllWindows()
        if args.out and recorded:
            print(f"{write_take(args.out, recorded)} kare -> {args.out}")


def run_cameras(args):
    """Hangi kamera indeksinin acildigini tarar.

    Sanal kameralar (iVCam gibi) her zaman 0'a dusmuyor; ayrica bagli
    degilken indeks var ama goruntu gelmiyor olabiliyor -- o yuzden sadece
    acilma degil, gercekten kare okunup okunmadigi da kontrol ediliyor.
    """
    import cv2

    print("kamera indeksleri taraniyor (0-5)...")
    found = 0
    for index in range(6):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            continue
        ok, frame = capture.read()
        if ok and frame is not None:
            height, width = frame.shape[:2]
            print(f"  --camera {index}  ACIK   {width}x{height}")
            found += 1
        else:
            print(f"  --camera {index}  BOS    acildi ama kare gelmiyor (kaynak bagli mi?)")
        capture.release()

    if not found:
        print("\nKullanilabilir kamera yok.")
        print("Bu makinede tek kamera e2eSoft iVCam (sanal): telefonda iVCam")
        print("uygulamasini acip PC istemcisine baglaman gerekiyor.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--profile",
        type=Path,
        help=(
            "kalibrasyon profili. DIKKAT: verilirse cekim dosyasina KALIBRE EDILMIS deger "
            "yazilir. Normalde bos birak -- bake operator'u kendi notr hesabini yapiyor, "
            "ham veri kaydetmek sonradan yeniden kalibre edebilmeyi saglar"
        ),
    )
    parser.add_argument("--filter", action="store_true", help="one-euro filtre uygula")
    sub = parser.add_subparsers(dest="command", required=True)

    video = sub.add_parser("video", help="video dosyasindan cekim uret")
    video.add_argument("--input", required=True, type=Path)
    video.add_argument("--out", required=True, type=Path)
    video.add_argument("--landmarks", action="store_true", help="478 ham landmark'i da yaz (cozucu icin; kareyi ~30 kat buyutur)")
    video.set_defaults(func=run_video)

    live = sub.add_parser("live", help="webcam'den canli yayin")
    live.add_argument("--udp", help="hedef, ornek 127.0.0.1:11111")
    live.add_argument("--out", type=Path, help="ayni anda dosyaya da kaydet")
    live.add_argument("--camera", type=int, default=0)
    live.add_argument("--preview", action="store_true")
    live.add_argument("--landmarks", action="store_true", help="478 ham landmark'i da yaz (cozucu icin; kareyi ~30 kat buyutur)")
    live.set_defaults(func=run_live)

    cameras = sub.add_parser("cameras", help="kullanilabilir kamera indekslerini tara")
    cameras.set_defaults(func=run_cameras)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
