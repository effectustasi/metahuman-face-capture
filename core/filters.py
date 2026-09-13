"""One-euro filtre. Stdlib disi bagimlilik yok, her iki taraftan da import edilir.

Kaynak: Casiez, Roussel, Vogel - "1 Euro Filter" (CHI 2012). Jitter'i
dusuk hizda bastirir, hizli hareketlerde gecikme eklemez -- yuz yakalamada
sabit dusuk-gecirgen filtreden bu yuzden iyi.
"""

from __future__ import annotations

import math


class LowPass:
    def __init__(self):
        self.value: float | None = None

    def __call__(self, x: float, alpha: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = alpha * x + (1.0 - alpha) * self.value
        return self.value


class OneEuroFilter:
    """min_cutoff dusuk -> daha cok yumusatma. beta yuksek -> hizli
    hareketlerde daha az gecikme."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = LowPass()
        self._dx = LowPass()
        self._last_time: float | None = None
        self._last_value: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, value: float, timestamp: float) -> float:
        if self._last_time is None or timestamp <= self._last_time:
            self._last_time = timestamp
            self._last_value = value
            self._x.value = value
            return value

        dt = timestamp - self._last_time
        derivative = (value - (self._last_value or value)) / dt
        edx = self._dx(derivative, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        filtered = self._x(value, self._alpha(cutoff, dt))

        self._last_time = timestamp
        self._last_value = value
        return filtered


class ChannelFilters:
    """Kanal basina bagimsiz one-euro, gruba gore farkli agresiflik.

    Tek bir filtre ayari tum yuze uymuyor:

    * **Goz kirpma** hizli ve kisa. MediaPipe'in blink'i zaten gec
      tetikleniyor; ustune agir filtre koyunca kirpma tamamen kayboluyor.
    * **Agiz** konusmada saniyede birkac kez sekil degistiriyor. Kaslara
      uygun yumusatma dudaklari lapa yapiyor, kelimeler okunmuyor --
      "dudaklar dogal degil" sikayetinin buyuk kismi burada.
    * **Kas ve yanak** yavas ve genis hareketler; burada agir yumusatma
      dogru, titremeyi bastiriyor.
    """

    BLINK_CHANNELS = frozenset({"eyeBlinkLeft", "eyeBlinkRight"})
    FAST_PREFIXES = ("mouth", "jaw")

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        blink_min_cutoff: float = 5.0,
        blink_beta: float = 0.05,
        mouth_min_cutoff: float = 3.0,
        mouth_beta: float = 0.02,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.blink_min_cutoff = blink_min_cutoff
        self.blink_beta = blink_beta
        self.mouth_min_cutoff = mouth_min_cutoff
        self.mouth_beta = mouth_beta
        self._filters: dict[str, OneEuroFilter] = {}

    def _get(self, name: str) -> OneEuroFilter:
        if name not in self._filters:
            if name in self.BLINK_CHANNELS:
                self._filters[name] = OneEuroFilter(self.blink_min_cutoff, self.blink_beta)
            elif name.startswith(self.FAST_PREFIXES):
                self._filters[name] = OneEuroFilter(self.mouth_min_cutoff, self.mouth_beta)
            else:
                self._filters[name] = OneEuroFilter(self.min_cutoff, self.beta)
        return self._filters[name]

    def __call__(self, blendshapes: dict[str, float], timestamp: float) -> dict[str, float]:
        return {name: self._get(name)(value, timestamp) for name, value in blendshapes.items()}
