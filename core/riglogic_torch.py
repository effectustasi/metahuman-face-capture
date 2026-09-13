"""RigLogic'in turevlenebilir yeniden yazimi (PyTorch).

Neden: numerik Jacobian her iterasyonda N+1 **sirali** RigLogic cagrisi
gerektiriyor ve RigLogic derlenmis bir CPU kutuphanesi -- GPU'ya
tasinamaz, paralellestirilemez. 174 parametrede iterasyon 341 ms.

Turevlenebilir yazarsak autograd tek geri gecisle tam gradyani veriyor:
N+1 carpani **tamamen kalkiyor**. Bu 2x degil, O(N) -> O(1) kazanci.

Zincir (API yoklanarak cikarildi, `scripts/05_extract_behavior.py`):

    GUI (174)
      |  parcali dogrusal:  raw = slope * gui + cut,  gui in [from, to]
    raw (263)
      |  PSD: ham kontrollerin agirlikli CARPIMLARI, raw'in ardina eklenir
    kontrol (808 = 263 + 545)
      |  122 joint grubu, her biri yogun (cikti x girdi) matris
    joint deltalari (7830 = 870 x 9)
      |  birebir gather
    blendshape agirliklari (782)

Dogruluk gercek RigLogic'e karsi dogrulanir:
`scripts/06_dump_reference.py` + `tests/test_riglogic_torch.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class TorchRigLogic(torch.nn.Module):
    """Kontrol vektorunden joint deltalari ve blendshape agirliklarina.

    Butun tablolar buffer olarak tutulur, boylece `.to("cuda")` hepsini
    birlikte tasir.
    """

    def __init__(
        self,
        data: dict,
        dtype: torch.dtype = torch.float32,
        dense_joints: bool = True,
    ):
        """dense_joints: 121 joint grubunu tek (7830 x 826) matriste birlestir.

        Olculdu -- gruplu halde her grup ayri bir kucuk matmul, yani 121
        kernel baslatmasi. Tek kare gradyaninda bu tamamen overhead:

            CPU  121 grup 11.40 ms -> tek GEMM  3.55 ms  (3.2x)
            CUDA 121 grup 37.70 ms -> tek GEMM  4.15 ms  (9.1x)

        Maliyeti 25 MB (matris %14.9 dolu). Batch buyudukce fark kapaniyor
        ama tek kare gercek zaman icin kritik.
        """
        super().__init__()

        self.gui_count = int(data["guiCount"])
        self.raw_count = int(data["rawCount"])
        self.psd_count = int(data["psdCount"])
        self.joint_count = int(data["jointCount"])
        self.blend_count = int(data["blendShapeChannelCount"])
        # Kontrol vektoru raw + PSD + RBF poz kontrolleri. RBF'ler joint
        # rotasyonlarindan suruluyor (addon'daki evaluate_rbfs yolu);
        # burada disaridan verilmezse sifir kalirlar.
        self.rbf_count = int(data.get("rbfPoseControlCount", 0))
        self.control_count = int(
            data.get("controlCount", self.raw_count + self.psd_count + self.rbf_count)
        )
        self.joint_output_count = self.joint_count * 9

        self.gui_control_names = [str(n) for n in data["guiControlNames"]]
        self.raw_control_names = [str(n) for n in data["rawControlNames"]]

        def register(name: str, array, dtype_override=None):
            tensor = torch.as_tensor(np.asarray(array), dtype=dtype_override or dtype)
            self.register_buffer(name, tensor, persistent=False)

        # --- GUI -> raw
        register("gui_in", data["guiToRawInputIndices"], torch.long)
        register("gui_out", data["guiToRawOutputIndices"], torch.long)
        register("gui_from", data["guiToRawFromValues"])
        register("gui_to", data["guiToRawToValues"])
        register("gui_slope", data["guiToRawSlopeValues"])
        register("gui_cut", data["guiToRawCutValues"])

        # --- PSD: satira gore grupla. Bir PSD ciktisi, kendi satirindaki
        # tum (deger * kontrol[sutun]) terimlerinin CARPIMI.
        rows = np.asarray(data["psdRowIndices"], dtype=np.int64)
        columns = np.asarray(data["psdColumnIndices"], dtype=np.int64)
        values = np.asarray(data["psdValues"], dtype=np.float64)

        # DIKKAT: `psdValues` bir CARPAN DEGIL.
        #
        # Once her terimi kendi degeriyle carptim; "hepsi +1" testi patladi
        # (PSD 453: sutun 191 deger 4.0, sutun 108 deger 1.0 -> benim 4.0,
        # RigLogic 1.0). Dort hipotez 32 gercek ornege karsi denendi:
        #
        #   carpim x deger      -> 3.0e+00 hata
        #   carpim / deger      -> 7.5e-01 hata
        #   carpim x ilk deger  -> 3.0e+00 hata
        #   SADECE carpim       -> 2.2e-08 hata   <-- dogru
        #
        # Yani PSD ciktisi refere edilen kontrollerin duz carpimi. Degerler
        # dizisi baska bir amaca hizmet ediyor (bu DNA'da sadece 1.0 ve 4.0
        # var). Farkli bir DNA'da anlami degisirse dogrulama testi yakalar.
        #
        # ayni terim sayisina sahip satirlari kova kova topla ki
        # vektorlestirilebilsin (olculdu: 545 PSD icin ~1430 terim)
        counts: dict[int, list[int]] = {}
        order = np.argsort(rows, kind="stable")
        rows_sorted = rows[order]
        unique_rows, starts, lengths = np.unique(
            rows_sorted, return_index=True, return_counts=True
        )
        for row, start, length in zip(unique_rows, starts, lengths):
            counts.setdefault(int(length), []).append(int(start))

        self._psd_buckets = []
        for length, start_list in sorted(counts.items()):
            indices = np.asarray(
                [order[start : start + length] for start in start_list], dtype=np.int64
            )
            bucket_rows = rows[indices[:, 0]]
            suffix = f"psd{length}"
            register(f"{suffix}_rows", bucket_rows, torch.long)
            register(f"{suffix}_cols", columns[indices], torch.long)
            # deger dizisi kasitli olarak KULLANILMIYOR, yukaridaki nota bak
            self._psd_buckets.append((length, suffix))

        # --- joint gruplari
        shapes = np.asarray(data["jointGroupShapes"], dtype=np.int64)
        value_offsets = np.asarray(data["jointGroupValueOffsets"], dtype=np.int64)
        input_offsets = np.asarray(data["jointGroupInputOffsets"], dtype=np.int64)
        output_offsets = np.asarray(data["jointGroupOutputOffsets"], dtype=np.int64)
        all_values = np.asarray(data["jointGroupValues"], dtype=np.float64)
        all_inputs = np.asarray(data["jointGroupInputIndices"], dtype=np.int64)
        all_outputs = np.asarray(data["jointGroupOutputIndices"], dtype=np.int64)

        self._joint_groups = []
        for group, (out_count, in_count) in enumerate(shapes):
            if out_count == 0 or in_count == 0:
                continue
            suffix = f"jg{group}"
            matrix = all_values[value_offsets[group] : value_offsets[group + 1]].reshape(
                out_count, in_count
            )
            register(f"{suffix}_m", matrix)
            register(
                f"{suffix}_in",
                all_inputs[input_offsets[group] : input_offsets[group + 1]],
                torch.long,
            )
            register(
                f"{suffix}_out",
                all_outputs[output_offsets[group] : output_offsets[group + 1]],
                torch.long,
            )
            self._joint_groups.append(suffix)

        # --- blendshape eslemesi
        register("blend_in", data["blendShapeInputIndices"], torch.long)
        register("blend_out", data["blendShapeOutputIndices"], torch.long)

        # --- istege bagli: gruplari tek yogun matrise indir
        self.dense_joints = dense_joints
        if dense_joints:
            merged = torch.zeros(self.joint_output_count, self.control_count, dtype=dtype)
            for suffix in self._joint_groups:
                matrix = getattr(self, f"{suffix}_m")
                inputs = getattr(self, f"{suffix}_in")
                outputs = getattr(self, f"{suffix}_out")
                merged[outputs[:, None], inputs[None, :]] += matrix
            self.register_buffer("joint_matrix", merged, persistent=False)

    @classmethod
    def load(cls, path: Path | str, **kwargs) -> "TorchRigLogic":
        with np.load(Path(path), allow_pickle=False) as data:
            return cls({key: data[key] for key in data.files}, **kwargs)

    # ------------------------------------------------------------------
    # katmanlar
    # ------------------------------------------------------------------

    def gui_to_raw(self, gui: torch.Tensor) -> torch.Tensor:
        """Parcali dogrusal. gui (B, 174) -> raw (B, 263).

        Bir satir yalnizca girdi kendi [from, to] araligindayken katki
        yapar. Maske sert; aralik disinda gradyan sifir olmali cunku
        kontrolun orada gercekten etkisi yok (ReLU gibi).
        """
        value = gui[:, self.gui_in]
        inside = (value >= self.gui_from) & (value <= self.gui_to)
        contribution = torch.where(
            inside, value * self.gui_slope + self.gui_cut, torch.zeros_like(value)
        )
        raw = gui.new_zeros((gui.shape[0], self.raw_count))
        return raw.index_add_(1, self.gui_out, contribution)

    def apply_psd(self, raw: torch.Tensor, rbf: torch.Tensor | None = None) -> torch.Tensor:
        """raw (B, 263) -> kontrol (B, 826). PSD'ler ham kontrollerin carpimi.

        rbf: (B, 18) RBF poz kontrolleri. Verilmezse sifir -- kafa/boyun
        rotasyonu surulmediginde RigLogic de sifir birakiyor.
        """
        controls = raw.new_zeros((raw.shape[0], self.control_count))
        controls[:, : self.raw_count] = raw
        if rbf is not None and self.rbf_count:
            start = self.raw_count + self.psd_count
            controls[:, start : start + self.rbf_count] = rbf

        for length, suffix in self._psd_buckets:
            rows = getattr(self, f"{suffix}_rows")
            cols = getattr(self, f"{suffix}_cols")
            # (B, n, length) -> carpim -> (B, n)
            gathered = controls[:, cols.reshape(-1)].reshape(raw.shape[0], -1, length)
            product = gathered.prod(dim=2)
            controls = controls.index_copy(1, rows, product)
        return controls

    def joint_deltas(self, controls: torch.Tensor) -> torch.Tensor:
        """kontrol (B, 826) -> joint deltalari (B, 7830)."""
        if self.dense_joints:
            return controls @ self.joint_matrix.T

        deltas = controls.new_zeros((controls.shape[0], self.joint_output_count))
        for suffix in self._joint_groups:
            matrix = getattr(self, f"{suffix}_m")
            inputs = getattr(self, f"{suffix}_in")
            outputs = getattr(self, f"{suffix}_out")
            result = controls[:, inputs] @ matrix.T
            deltas = deltas.index_add(1, outputs, result)
        return deltas

    def blend_weights(self, controls: torch.Tensor) -> torch.Tensor:
        """kontrol (B, 808) -> blendshape agirliklari (B, 782)."""
        weights = controls.new_zeros((controls.shape[0], self.blend_count))
        return weights.index_copy(1, self.blend_out, controls[:, self.blend_in])

    def forward(
        self, gui: torch.Tensor, rbf: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if gui.dim() == 1:
            gui = gui.unsqueeze(0)
        controls = self.apply_psd(self.gui_to_raw(gui), rbf)
        return self.joint_deltas(controls), self.blend_weights(controls)
