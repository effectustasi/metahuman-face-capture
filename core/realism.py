"""ARKit'in kanali OLMAYAN kontrolleri turetir.

Olculdu: DNA'da 23 goz kontrolu var, ARKit'ten gelen tabloyla sadece 8'i
suruluyor. Kalan 15'i (goz kapaklari, kapak baskisi, goz bebegi) ARKit'in
52 kanalinda **karsiligi olmadigi icin** olu duruyor. Bunlar olculemez ama
turetilebilir -- gozun anatomisi bunlari bakis yonune ve kisilmaya baglar.

Katman **eslemeden SONRA** calisir: girdisi GUI eksen degerleri, ciktisi
ek GUI eksenleri. Boylece kaynagi (webcam / iPhone / replay) umursamiyor.

Yonler DNA'dan dogrulandi, ezberden degil:

    CTRL_*_eye.ty          + yukari bak      - asagi bak
    CTRL_*_eye_eyelidU.ty  + kapak gevser    - ust kapak KALKAR
    CTRL_*_eye_eyelidD.ty  + alt kapak kalkar - alt kapak iner
    CTRL_*_eye_lidPress.ty + kapak baskisi
"""

from __future__ import annotations

import math
import random

# Agiz: buzusturmede dudaklari birbirine getiren eksenler.
# Yon DNA'dan dogrulandi -- hepsi [0..1], pozitif = dudaklar birlesir.
LIPS_TOGETHER_AXES = (
    "CTRL_L_mouth_lipsTogetherU.y",
    "CTRL_R_mouth_lipsTogetherU.y",
    "CTRL_L_mouth_lipsTogetherD.y",
    "CTRL_R_mouth_lipsTogetherD.y",
)
PUCKER_AXES = ("CTRL_L_mouth_purseU.y", "CTRL_R_mouth_purseU.y")
JAW_AXIS = "CTRL_C_jaw.y"

# Bakis ekseni -> turetilecek kapak eksenleri. Isaretler DNA dogrulamasindan.
EYE_SIDES = (
    {
        "gaze": "CTRL_L_eye.y",
        "upper": "CTRL_L_eye_eyelidU.y",
        "lower": "CTRL_L_eye_eyelidD.y",
        "press": "CTRL_L_eye_lidPress.y",
        "squint": "CTRL_L_eye_squintInner.y",
        "blink": "CTRL_L_eye_blink.y",
        "gaze_x": "CTRL_L_eye.x",
    },
    {
        "gaze": "CTRL_R_eye.y",
        "upper": "CTRL_R_eye_eyelidU.y",
        "lower": "CTRL_R_eye_eyelidD.y",
        "press": "CTRL_R_eye_lidPress.y",
        "squint": "CTRL_R_eye_squintInner.y",
        "blink": "CTRL_R_eye_blink.y",
        "gaze_x": "CTRL_R_eye.x",
    },
)


class MicroSaccades:
    """Gozun asla sabit durmamasi.

    Gercek goz sabit bakarken bile saniyede birkac kez kucuk siciramalar
    yapar (mikro-sakkad). Tamamen sabit bir bakis olu gorunuyor -- bu
    katmanin en ucuz, en fark edilir kazanci.

    Duz gurultu degil: sakkad "tut ve sicra" karakterinde. Araya yavas
    suruklenme (drift) eklenir.
    """

    def __init__(
        self,
        amplitude: float = 0.02,
        interval: float = 0.4,
        interval_jitter: float = 0.25,
        seed: int | None = None,
    ):
        self.amplitude = amplitude
        self.interval = interval
        self.interval_jitter = interval_jitter
        self._random = random.Random(seed)
        self._target = (0.0, 0.0)
        self._current = (0.0, 0.0)
        self._next_time = 0.0

    def reset(self) -> None:
        self._target = (0.0, 0.0)
        self._current = (0.0, 0.0)
        self._next_time = 0.0

    def offset(self, timestamp: float) -> tuple[float, float]:
        if timestamp >= self._next_time:
            self._target = (
                self._random.gauss(0.0, self.amplitude),
                self._random.gauss(0.0, self.amplitude * 0.6),  # dikeyde daha az
            )
            spread = self.interval * self.interval_jitter
            self._next_time = timestamp + max(0.05, self._random.gauss(self.interval, spread))

        # sakkad hizli: hedefe kisa surede yaklas (yaklasik 40 ms)
        blend = 0.35
        self._current = (
            self._current[0] + (self._target[0] - self._current[0]) * blend,
            self._current[1] + (self._target[1] - self._current[1]) * blend,
        )
        return self._current


class RealismLayer:
    """ARKit'in ulasamadigi goz kontrollerini turetir."""

    def __init__(
        self,
        eyelid_follow: float = 0.85,
        lower_lid_follow: float = 0.25,
        lid_press: float = 0.5,
        saccades: float = 1.0,
        saccade_amplitude: float = 0.02,
        pucker_close: float = 0.7,
        seed: int | None = None,
    ):
        self.eyelid_follow = eyelid_follow
        self.lower_lid_follow = lower_lid_follow
        self.lid_press = lid_press
        self.saccades = saccades
        self.pucker_close = pucker_close
        self._saccades = MicroSaccades(amplitude=saccade_amplitude, seed=seed)

    def reset(self) -> None:
        self._saccades.reset()

    def extra_axes(self, axes: dict[str, float], timestamp: float = 0.0) -> dict[str, float]:
        """Esleme ciktisina eklenecek turetilmis eksenler.

        `axes` degistirilmez; donen sozluk cagrian tarafindan birlestirilir.
        """
        extra: dict[str, float] = {}

        saccade = self._saccades.offset(timestamp) if self.saccades > 0.0 else (0.0, 0.0)

        # --- buzusturmede dudaklar birlesir
        #
        # Sorun: Epic'in tablosunda mouthPucker `funnel` kontrollerini 0.75
        # agirlikla suruyor ve funnel agzi "O" seklinde ACIYOR. MediaPipe
        # opucukte hem mouthPucker hem mouthFunnel bildiriyor, ikisi ayni
        # eksende toplanip doyuma gidiyor -> agiz genis bir O oluyor ve
        # ORTADA BOSLUK kaliyor.
        #
        # Dudaklari kapatan `lipsTogether` kontrollerini ise sadece
        # mouthClose suruyor, o da buzusturmede tetiklenmiyor.
        #
        # Anatomik olarak buzusturme (orbicularis oris kasilmasi) dudaklari
        # hem one cikarir hem BIRLESTIRIR. Cene acikken birlesemezler --
        # gozdeki "kirpma sonurmesi" ile ayni kalip.
        if self.pucker_close:
            pucker = max((axes.get(key, 0.0) for key in PUCKER_AXES), default=0.0)
            if pucker > 0.0:
                jaw = max(0.0, axes.get(JAW_AXIS, 0.0))
                amount = pucker * self.pucker_close * max(0.0, 1.0 - jaw)
                if amount > 0.0:
                    for key in LIPS_TOGETHER_AXES:
                        extra[key] = max(axes.get(key, 0.0), amount)

        for side in EYE_SIDES:
            gaze = axes.get(side["gaze"], 0.0)
            blink = axes.get(side["blink"], 0.0)

            # --- kapak bakisi takip eder
            #
            # Ust kapak dikey bakisi neredeyse birebir izler: asagi bakinca
            # iner, yukari bakinca kalkar. DNA'da eyelidU'nun NEGATIFI kapagi
            # kaldiriyor, o yuzden isaret ters.
            #
            # Goz kirpik/kapaliyken takip anlamsiz, hatta kapagi geri aciyor
            # -- kirpma oraninda soneriyoruz.
            attenuation = max(0.0, 1.0 - blink)
            if self.eyelid_follow:
                extra[side["upper"]] = -gaze * self.eyelid_follow * attenuation
            if self.lower_lid_follow:
                # alt kapak ayni yonde ama cok daha az
                extra[side["lower"]] = gaze * self.lower_lid_follow * attenuation

            # --- kisilma kapak baskisi yaratir
            if self.lid_press:
                squint = axes.get(side["squint"], 0.0)
                if squint:
                    extra[side["press"]] = squint * self.lid_press

            # --- mikro-sakkadlar
            if self.saccades > 0.0:
                key_x, key_y = side["gaze_x"], side["gaze"]
                extra[key_x] = axes.get(key_x, 0.0) + saccade[0] * self.saccades
                extra[key_y] = gaze + saccade[1] * self.saccades

        return extra


def merge(axes: dict[str, float], extra: dict[str, float], limits: dict | None = None) -> dict[str, float]:
    """Turetilmis eksenleri esleme ciktisiyla birlestir ve kirp.

    Turetilmis degerler var olanin YERINE gecer (uzerine eklenmez):
    kapak eksenlerini zaten kimse surmuyordu, bakis eksenini de sakkad
    zaten mevcut degeri iceriyor.
    """
    merged = dict(axes)
    merged.update(extra)

    if limits:
        for key, value in merged.items():
            limit = limits.get(key)
            if limit:
                merged[key] = max(limit["min"], min(limit["max"], value))
    return merged


def blink_curve(value: float, gamma: float = 1.0) -> float:
    """Kirpma egrisini sekillendirir. gamma>1 gec kapanip hizli kapatir.

    Olculdu: ARKit'in kirpma ZAMANLAMASI zaten dogru (kapanma 89 ms,
    acilma 194 ms, oran 2.19 -- literaturle uyumlu). Yani egriyi yeniden
    sekillendirmeye gerek yok; bu fonksiyon sadece gerekirse diye duruyor
    ve varsayilani kimlik.
    """
    if gamma == 1.0:
        return value
    return math.copysign(abs(value) ** gamma, value)
