"""MediaPipe landmark'larini karakterin uzayina tasir.

Neden ayri bir katman: gozlem SENIN yuzun, model ICARDI'nin yuzu. Ham
konumlari karsilastirirsak cozucu ifade kontrollerini kimlik farkini
kapatmak icin harcar -- burun daha genisse "burun genislet" kontrolunu
sonuna kadar acar ve ifade bilgisi kaybolur.

Care: iki tarafi da KENDI notrune gore okumak. Kimlik farki notrde
duruyor, iki taraftan da cikarilinca birinci mertebeden iptal oluyor;
geriye sadece hareket kaliyor.

    hedef = karakter_notru + s*R @ (gozlem_hizali - gozlem_notru)

`s` ve `R` bir kez notrler arasindan olculuyor (kafa boyu/yonu farki),
her karede degil.

Iki tuzak:

1. MediaPipe x'i GENISLIGE, y'yi YUKSEKLIGE bolerek normalize ediyor.
   Kare olmayan goruntude ayni normalize fark farkli piksel demektir;
   duzeltilmezse yuz dikey ezik okunur. `prepare` bunu duzeltiyor.

2. z metrik degil. MediaPipe belgeleri "kabaca x ile ayni olcek" diyor
   ama garanti etmiyor. Gercek cekimle olculene kadar z'ye tam guvenme.

`learn_rigid_subset` susleme degil, ZORUNLU. Kafa pozunu tum yuzle
ayiklarsan agiz acilinca cenenin hareketi kismen "kafa dondu" diye
okunur. Sentetik olcum (tests/test_observation.py):

    bolgesel ifade, tum yuz Procrustes   -> hareketin %11.3'u kayboluyor
    bolgesel ifade, rijit altkume        -> %0

Bilinen sinir: her landmark bagimsiz oynarsa (%25 kayip) hareket rijit
donusumden ayirt edilemez. Gercek yuzlerde olmuyor cunku ifadeler
bolgesel.
"""

from __future__ import annotations

import numpy as np


def prepare(landmarks, resolution, landmark_ids=None) -> np.ndarray:
    """Ham MediaPipe cikisini izotropik birime cevirir.

    Cikis birimi "goruntu genisligi = 1"; boylece x, y, z ayni olcekte.
    """
    points = np.asarray(landmarks, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"landmark dizisi (N, 3) olmali, {points.shape} geldi")
    width, height = float(resolution[0]), float(resolution[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"gecersiz cozunurluk: {resolution}")

    scaled = points.copy()
    scaled[:, 1] *= height / width  # y'yi x ile ayni olcege getir
    if landmark_ids is not None:
        scaled = scaled[np.asarray(landmark_ids, dtype=int)]
    return scaled


def procrustes(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """source -> target icin (olcek, rotasyon, oteleme). Yansima uretmez."""
    source_center, target_center = source.mean(0), target.mean(0)
    a, b = source - source_center, target - target_center
    u, s, vt = np.linalg.svd(a.T @ b)
    correction = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(vt.T @ u.T)))])
    rotation = vt.T @ correction @ u.T
    scale = float((s * np.diag(correction)).sum() / max((a * a).sum(), 1e-12))
    return scale, rotation, target_center - scale * (rotation @ source_center)


class ObservationSpace:
    """Gozlem karelerini karakterin DNA uzayina (cm) tasir."""

    def __init__(self, model_neutral: np.ndarray, landmark_ids):
        """`model_neutral`: karakterin notr landmark konumlari (L, 3), DNA cm.
        `landmark_ids`: bu L landmark'in MediaPipe'daki 0-477 indeksleri --
        cozucu bozuk karsiliklari attigi icin 478'in hepsi olmayabilir."""
        self.model_neutral = np.asarray(model_neutral, dtype=np.float64)
        self.landmark_ids = np.asarray(landmark_ids, dtype=int)
        if self.model_neutral.shape != (len(self.landmark_ids), 3):
            raise ValueError(
                f"model notru {self.model_neutral.shape}, landmark sayisi {len(self.landmark_ids)}"
            )
        self.neutral: np.ndarray | None = None
        self.scale: float | None = None
        self.rotation: np.ndarray | None = None
        self.rigid_mask: np.ndarray | None = None

    # -- notr kurulumu ----------------------------------------------------

    def learn_neutral(self, frames) -> int:
        """`frames`: (landmarks, resolution) ciftleri. Ifadesiz olmali.

        Ortalama aliniyor: tek kare MediaPipe titremesini de icine alir,
        notr en cok guvenilmesi gereken sey oldugu icin ucuz olan yol degil
        dogru olan yol tercih edildi."""
        stack = [prepare(lm, res, self.landmark_ids) for lm, res in frames]
        if not stack:
            raise ValueError("notr icin en az bir kare gerekiyor")
        self.neutral = np.mean(stack, axis=0)
        self.scale, self.rotation, _ = procrustes(self.neutral, self.model_neutral)
        return len(stack)

    # -- rijit altkume ----------------------------------------------------

    def learn_rigid_subset(self, frames, keep: float = 0.4) -> np.ndarray:
        """Kafa pozunu ayiklarken hangi landmark'lara guvenilecegini OLCER.

        Tum yuzle Procrustes yapmak agiz acilinca cenenin hareketini kafa
        donusu sanar. Ifadeyle en az kipirdayan landmark'lar rijittir --
        bunu tahmin etmek yerine cekimden olcuyoruz.
        """
        if self.neutral is None:
            raise RuntimeError("once learn_neutral cagir")
        stack = np.stack([prepare(lm, res, self.landmark_ids) for lm, res in frames])
        # her kareyi tum yuzle kabaca hizala, sonra kalan oynakligi olc
        aligned = []
        for points in stack:
            s, r, t = procrustes(points, self.neutral)
            aligned.append(s * (r @ points.T).T + t)
        motion = np.linalg.norm(np.stack(aligned) - self.neutral, axis=2).mean(axis=0)
        count = max(8, int(len(motion) * keep))
        self.rigid_mask = np.argsort(motion)[:count]
        return self.rigid_mask

    # -- kare donusumu ----------------------------------------------------

    def to_character(self, landmarks, resolution) -> np.ndarray:
        """Bir kareyi karakterin uzayinda hedef landmark konumlarina cevirir."""
        if self.neutral is None or self.rotation is None:
            raise RuntimeError("once learn_neutral cagir")
        points = prepare(landmarks, resolution, self.landmark_ids)

        # 1) kafa pozunu ayikla -- gozlemi kendi notrune oturt
        subset = self.rigid_mask if self.rigid_mask is not None else slice(None)
        s, r, t = procrustes(points[subset], self.neutral[subset])
        aligned = s * (r @ points.T).T + t

        # 2) yer degistirmeyi karakterin olcegine tasi ve notrune ekle
        displacement = aligned - self.neutral
        return self.model_neutral + self.scale * (self.rotation @ displacement.T).T
