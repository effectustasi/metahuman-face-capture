"""facecap arayuzu -- tek pencereden kamera, canli yayin ve kayit.

Uc ayri .bat dosyasinin yerini alir. Onlarin sorunu calismamalari degildi;
sorun geri bildirim vermemeleriydi. Kanal barlari olmadan "jawOpen gercekten
tetikleniyor mu" sorusunun cevabi yok, kalibrasyon korlemesine yapiliyordu.

detector/.venv ile calisir (mediapipe + cv2 + PIL orada):

    detector\\.venv\\Scripts\\pythonw.exe gui.py

MediaPipe ayri bir is parcaciginda kosar, Tk ana parcacikta. cv2.imshow
KULLANILMIYOR -- HighGUI kendi olay dongusunu isletiyor ve Tk mainloop ile
yan yana Windows'ta kilitlenebiliyor. Onizleme dogrudan Tk'ye ciziliyor.
"""

from __future__ import annotations

import json
import queue
import socket
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "detector"))

import detect  # noqa: E402
from core.filters import ChannelFilters  # noqa: E402
from core.take import Frame, write_take  # noqa: E402

PREVIEW_WIDTH = 400

# Bar gruplari. Tek bir "en yuksek 8" listesi calismiyor: MediaPipe notr
# yuzde bile goz/kas kanallarini yuksek veriyor ve liste kalici olarak
# onlarla doluyor. Gercek kullanicinin ifadesiz 30 karesinde olculdu:
#
#   eyeSquintLeft 0.473   browDownRight 0.395   browDownLeft 0.305
#   eyeLookUpLeft 0.301   eyeBlinkLeft  0.290   eyeSquintRight 0.267
#
# Tam sekiz kanal surekli 0.2 uzerinde. Agiz kanallari notrde 0.00 olmasina
# ragmen listeye hic giremiyordu. Bu yuzden: bolge bolge, ve HAM deger degil
# NOTRDEN SAPMA gosteriliyor.
BAR_GROUPS = (
    ("Agiz / cene", ("mouth", "jaw"), 4),
    ("Goz", ("eye",), 3),
    ("Kas / burun", ("brow", "nose", "cheek"), 2),
)
BASELINE_FRAMES = 30


class Capture:
    """Yakalama is parcacigi. UI ile tek bir kuyruk uzerinden konusur."""

    def __init__(self, settings: dict, events: queue.Queue):
        self.settings = settings
        self.events = events
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.recorded: list[Frame] = []
        self.packets = 0

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_flag.set()

    def _emit(self, kind: str, payload=None) -> None:
        self.events.put((kind, payload))

    def _run(self) -> None:
        try:
            self._loop()
        except Exception as error:  # noqa: BLE001 -- UI'ye tasinmasi sart
            self._emit("error", f"{type(error).__name__}: {error}")
        finally:
            self._emit("stopped", None)

    def _loop(self) -> None:
        import cv2

        mp = detect.import_mediapipe()
        settings = self.settings

        sender = None
        address = None
        if settings["udp"]:
            sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            address = (settings["host"], settings["port"])

        filters = ChannelFilters() if settings["filter"] else None
        state = {"index": 0, "resolution": None}
        start_time = time.time()

        def on_result(result, output_image, timestamp_ms):
            blendshapes = detect.extract_blendshapes(result)
            if not blendshapes:
                self._emit("face", False)
                return
            state["index"] += 1
            t = timestamp_ms / 1000.0
            if filters:
                blendshapes = filters(blendshapes, t)

            frame = Frame(
                t=t,
                frame=state["index"],
                bs=blendshapes,
                head=detect.extract_head(result),
                lm=detect.extract_landmarks(result) if settings["landmarks"] else None,
                res=state["resolution"] if settings["landmarks"] else None,
            )
            if sender is not None:
                sender.sendto(
                    json.dumps(frame.to_json(), separators=(",", ":")).encode("utf-8"),
                    address,
                )
                self.packets += 1
            if settings["record"]:
                self.recorded.append(frame)
            self._emit("frame", (blendshapes, self.packets, state["index"]))

        landmarker = detect.build_landmarker(
            mp, settings["model"], "LIVE_STREAM", callback=on_result
        )
        capture = cv2.VideoCapture(settings["camera"], cv2.CAP_DSHOW)
        if not capture.isOpened():
            raise RuntimeError(f"kamera {settings['camera']} acilamadi")

        self._emit("started", None)
        last_timestamp = -1
        last_preview = 0.0
        try:
            while not self.stop_flag.is_set():
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError("kameradan kare gelmiyor")
                state["resolution"] = [image.shape[1], image.shape[0]]

                # detect_async zaman damgasinin KESIN artmasini sart kosuyor;
                # ayni milisaniyede iki kare gelirse surec oluyor.
                timestamp = int((time.time() - start_time) * 1000)
                if timestamp <= last_timestamp:
                    timestamp = last_timestamp + 1
                last_timestamp = timestamp

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                landmarker.detect_async(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp
                )

                # Onizlemeyi 15 fps'e kis: her kareyi PhotoImage'a cevirmek
                # yakalamanin kendisinden pahali, kuyrugu sisiriyor.
                now = time.time()
                if settings["preview"] and now - last_preview > 1 / 15:
                    last_preview = now
                    height = int(rgb.shape[0] * PREVIEW_WIDTH / rgb.shape[1])
                    self._emit("preview", cv2.resize(rgb, (PREVIEW_WIDTH, height)))
        finally:
            capture.release()
            landmarker.close()
            if sender is not None:
                sender.close()
            if settings["record"] and self.recorded:
                count = write_take(settings["record"], self.recorded)
                self._emit("saved", f"{count} kare kaydedildi")


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("facecap")
        root.minsize(780, 560)

        self.events: queue.Queue = queue.Queue()
        self.capture: Capture | None = None
        self.photo = None
        self.frame_times: list[float] = []
        self.baseline: dict[str, float] | None = None
        self.baseline_samples: list[dict] = []

        self._build()
        self.root.after(50, self._pump)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.scan_cameras()

    # -- arayuz ----------------------------------------------------------

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        top = ttk.Frame(outer)
        top.pack(fill="both", expand=True)

        left = ttk.LabelFrame(top, text="Onizleme", padding=6)
        left.pack(side="left", fill="both", expand=True)
        self.preview = ttk.Label(left, anchor="center", text="\n\n\nkapali")
        self.preview.pack(fill="both", expand=True)

        right = ttk.Frame(top, padding=(10, 0, 0, 0))
        right.pack(side="left", fill="y")

        box = ttk.LabelFrame(right, text="Kaynak", padding=8)
        box.pack(fill="x")
        row = ttk.Frame(box)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Kamera").pack(side="left")
        self.camera = ttk.Combobox(row, width=6, state="readonly", values=["0"])
        self.camera.set("0")
        self.camera.pack(side="left", padx=4)
        self.scan_button = ttk.Button(row, text="Tara", width=6, command=self.scan_cameras)
        self.scan_button.pack(side="left")

        box2 = ttk.LabelFrame(right, text="Blender'a gonder", padding=8)
        box2.pack(fill="x", pady=(8, 0))
        self.udp = tk.BooleanVar(value=True)
        ttk.Checkbutton(box2, text="UDP ile gonder", variable=self.udp).pack(anchor="w")
        row = ttk.Frame(box2)
        row.pack(fill="x", pady=2)
        self.host = ttk.Entry(row, width=13)
        self.host.insert(0, "127.0.0.1")
        self.host.pack(side="left")
        ttk.Label(row, text=":").pack(side="left")
        self.port = ttk.Entry(row, width=6)
        self.port.insert(0, "11111")
        self.port.pack(side="left")

        box3 = ttk.LabelFrame(right, text="Secenekler", padding=8)
        box3.pack(fill="x", pady=(8, 0))
        self.preview_on = tk.BooleanVar(value=True)
        self.filter_on = tk.BooleanVar(value=True)
        self.landmarks_on = tk.BooleanVar(value=False)
        self.record_on = tk.BooleanVar(value=False)
        ttk.Checkbutton(box3, text="Onizleme", variable=self.preview_on).pack(anchor="w")
        ttk.Checkbutton(box3, text="Yumusatma filtresi", variable=self.filter_on).pack(anchor="w")
        ttk.Checkbutton(
            box3, text="Ham landmark (cozucu icin)", variable=self.landmarks_on
        ).pack(anchor="w")
        ttk.Checkbutton(
            box3, text="Dosyaya kaydet", variable=self.record_on, command=self._toggle_record
        ).pack(anchor="w")
        row = ttk.Frame(box3)
        row.pack(fill="x", pady=(2, 0))
        self.record_path = ttk.Entry(row, width=17)
        self.record_path.insert(0, str(ROOT / "takes" / "cekim.jsonl"))
        self.record_path.pack(side="left")
        self.browse = ttk.Button(row, text="...", width=3, command=self._pick_file)
        self.browse.pack(side="left", padx=(3, 0))
        self._toggle_record()

        self.start_button = tk.Button(
            right,
            text="BASLAT",
            font=("Segoe UI", 13, "bold"),
            bg="#2e7d32",
            fg="white",
            activebackground="#1b5e20",
            activeforeground="white",
            relief="flat",
            command=self.toggle,
        )
        self.start_button.pack(fill="x", pady=(12, 6), ipady=8)

        self.status = ttk.Label(right, text="hazir", foreground="#555555", wraplength=190)
        self.status.pack(anchor="w")

        bottom = ttk.LabelFrame(outer, text="Kanallar (notrden sapma)", padding=8)
        bottom.pack(fill="x", pady=(10, 0))

        header = ttk.Frame(bottom)
        header.pack(fill="x")
        self.baseline_label = ttk.Label(header, text="", foreground="#ef6c00")
        self.baseline_label.pack(side="left")
        self.reset_baseline = ttk.Button(
            header, text="Notru Yeniden Al", command=self._reset_baseline, state="disabled"
        )
        self.reset_baseline.pack(side="right")

        self.bars = {}
        for title, _, count in BAR_GROUPS:
            group = ttk.Frame(bottom)
            group.pack(fill="x", pady=(6, 0))
            ttk.Label(group, text=title, font=("Segoe UI", 8, "bold"), foreground="#666666").pack(
                anchor="w"
            )
            rows = []
            for _ in range(count):
                row = ttk.Frame(group)
                row.pack(fill="x")
                name = ttk.Label(row, text="", width=24, anchor="w")
                name.pack(side="left")
                meter = ttk.Progressbar(row, maximum=1.0, length=320)
                meter.pack(side="left", padx=6)
                value = ttk.Label(row, text="", width=11, anchor="e")
                value.pack(side="left")
                rows.append((name, meter, value))
            self.bars[title] = rows

    def _toggle_record(self) -> None:
        state = "normal" if self.record_on.get() else "disabled"
        self.record_path.configure(state=state)
        self.browse.configure(state=state)

    def _pick_file(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".jsonl", initialdir=str(ROOT / "takes")
        )
        if path:
            self.record_path.delete(0, "end")
            self.record_path.insert(0, path)

    # -- kamera taramasi -------------------------------------------------

    def scan_cameras(self) -> None:
        if self.capture is not None:
            return  # yayin sirasinda tarama kamerayi kilitler
        self.scan_button.configure(state="disabled", text="...")

        def work():
            import cv2

            found = []
            for index in range(6):
                probe = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                if probe.isOpened() and probe.read()[0]:
                    found.append(str(index))
                probe.release()
            self.events.put(("cameras", found))

        threading.Thread(target=work, daemon=True).start()

    # -- baslat / durdur -------------------------------------------------

    def toggle(self) -> None:
        if self.capture is not None:
            self.status.configure(text="durduruluyor...", foreground="#555555")
            self.start_button.configure(state="disabled")
            self.capture.stop()
            return

        try:
            port = int(self.port.get())
        except ValueError:
            self.status.configure(text="port bir sayi olmali", foreground="#c62828")
            return

        settings = {
            "camera": int(self.camera.get()),
            "udp": self.udp.get(),
            "host": self.host.get().strip(),
            "port": port,
            "preview": self.preview_on.get(),
            "filter": self.filter_on.get(),
            "landmarks": self.landmarks_on.get(),
            "record": Path(self.record_path.get()) if self.record_on.get() else None,
            "model": detect.DEFAULT_MODEL,
        }
        self.frame_times.clear()
        self._reset_baseline()
        self.baseline_label.configure(text="notr ogreniliyor...", foreground="#ef6c00")
        self.reset_baseline.configure(state="disabled")
        self.capture = Capture(settings, self.events)
        self.start_button.configure(text="baslatiliyor...", state="disabled")
        self.status.configure(text="model yukleniyor", foreground="#555555")
        self.capture.start()

    # -- olay dongusu ----------------------------------------------------

    def _pump(self) -> None:
        latest_frame = None
        latest_preview = None
        face_lost = False
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "frame":
                    latest_frame = payload
                elif kind == "preview":
                    latest_preview = payload
                elif kind == "face":
                    face_lost = True
                elif kind == "cameras":
                    self._on_cameras(payload)
                elif kind == "started":
                    self.start_button.configure(text="DURDUR", bg="#c62828", state="normal")
                    self.status.configure(text="calisiyor", foreground="#2e7d32")
                elif kind == "stopped":
                    self._on_stopped()
                elif kind == "saved":
                    self.status.configure(text=payload, foreground="#2e7d32")
                elif kind == "error":
                    self.status.configure(text=payload, foreground="#c62828")
        except queue.Empty:
            pass

        if latest_preview is not None:
            self._draw(latest_preview)
        if latest_frame is not None:
            self._update_channels(*latest_frame)
        elif face_lost and self.capture is not None:
            self.status.configure(text="yuz bulunamiyor", foreground="#ef6c00")
        self.root.after(40, self._pump)

    def _on_stopped(self) -> None:
        self.capture = None
        self.start_button.configure(text="BASLAT", bg="#2e7d32", state="normal")
        self.preview.configure(image="", text="\n\n\nkapali")
        self.photo = None
        for rows in self.bars.values():
            for name, meter, value in rows:
                name.configure(text="")
                meter["value"] = 0
                value.configure(text="")
        self.baseline_label.configure(text="")
        self.reset_baseline.configure(state="disabled")

    def _on_cameras(self, found) -> None:
        self.scan_button.configure(state="normal", text="Tara")
        if found:
            self.camera.configure(values=found)
            if self.camera.get() not in found:
                self.camera.set(found[0])
        else:
            self.status.configure(text="kamera bulunamadi", foreground="#c62828")

    def _draw(self, rgb) -> None:
        from PIL import Image, ImageTk

        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.preview.configure(image=self.photo, text="")

    def _reset_baseline(self) -> None:
        self.baseline = None
        self.baseline_samples = []

    def _update_channels(self, blendshapes, packets, frames) -> None:
        now = time.time()
        self.frame_times.append(now)
        self.frame_times = [t for t in self.frame_times if now - t < 2.0]
        fps = len(self.frame_times) / 2.0

        # notr taban: ilk 30 karenin ortalamasi. Tek kare yeterli degil,
        # MediaPipe kare kare titriyor ve taban ifadeye gore kayardi.
        if self.baseline is None:
            self.baseline_samples.append(blendshapes)
            remaining = BASELINE_FRAMES - len(self.baseline_samples)
            if remaining > 0:
                self.baseline_label.configure(
                    text=f"notr ogreniliyor, ifadesiz dur ({remaining})", foreground="#ef6c00"
                )
            else:
                keys = {k for s in self.baseline_samples for k in s}
                self.baseline = {
                    k: sum(s.get(k, 0.0) for s in self.baseline_samples)
                    / len(self.baseline_samples)
                    for k in keys
                }
                self.baseline_label.configure(text="notr alindi", foreground="#2e7d32")
                self.reset_baseline.configure(state="normal")

        base = self.baseline or {}
        for title, prefixes, count in BAR_GROUPS:
            picked = sorted(
                (
                    (max(0.0, score - base.get(channel, 0.0)), score, channel)
                    for channel, score in blendshapes.items()
                    if any(p in channel.lower() for p in prefixes)
                ),
                reverse=True,
            )[:count]
            rows = self.bars[title]
            padded = picked + [(0.0, 0.0, "")] * (count - len(picked))
            for (name, meter, value), (delta, raw, channel) in zip(rows, padded):
                name.configure(text=channel)
                meter["value"] = delta
                # hem sapma hem ham deger: sapma "oynadi mi", ham deger
                # "kanal doyuma gitti mi" sorusunu cevapliyor
                value.configure(text=f"{delta:.2f} ({raw:.2f})" if channel else "")

        self.status.configure(
            text=f"yuz VAR | {fps:.0f} fps | {frames} kare | {packets} paket",
            foreground="#2e7d32",
        )

    def _on_close(self) -> None:
        if self.capture is not None:
            self.capture.stop()
            self.root.after(500, self.root.destroy)
        else:
            self.root.destroy()


def main() -> None:
    # pythonw ile acilinca konsol yok; acilis sirasindaki bir hata (eksik
    # model dosyasi, bozuk venv) sessizce kaybolur ve pencere hic gelmez.
    # Kullanici "tikladim, bir sey olmadi" der. Onun yerine kutu goster.
    try:
        root = tk.Tk()
        for theme in ("vista", "clam"):
            try:
                ttk.Style().theme_use(theme)
                break
            except tk.TclError:
                continue
        App(root)
    except Exception as error:  # noqa: BLE001
        import traceback
        from tkinter import messagebox

        messagebox.showerror("facecap acilamadi", traceback.format_exc())
        raise SystemExit(1) from error
    root.mainloop()


if __name__ == "__main__":
    main()
