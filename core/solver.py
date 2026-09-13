"""Gozlemden rig kontrollerini cozer (analysis-by-synthesis).

Zincir tamamen turevlenebilir:

    GUI (174)
      -> TorchRigLogic      -> joint deltalari + blendshape agirliklari
      -> TorchFaceModel     -> vertex konumlari
      -> baryantrik         -> landmark konumlari
      -> kayip              vs gozlenen landmark'lar

Autograd tam gradyani tek geri gecisle veriyor; numerik Jacobian'in
gerektirdigi 175 sirali RigLogic cagrisi tamamen kalkiyor (341 ms -> 4.5 ms).

**Gozlem hizalamasi:** MediaPipe landmark'lari kendi normalize uzayinda
geliyor ve kafa pozunu iceriyor. Kamera ic parametrelerini bilmedigimiz
icin 2D yeniden izdusum yerine once RIJIT HIZALAMA yapiyoruz (Procrustes:
donme + oteleme + tek olcek). Boylece kafa pozu ve olcek ayiklanip geriye
sadece IFADE kaliyor -- cozucunun aradigi sey de bu.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .facemodel import JOINT_STRIDE, SCALE_FACTOR  # noqa: F401  (SCALE_FACTOR belgeleme icin)
from .riglogic_torch import TorchRigLogic


def euler_xyz_to_matrix(angles: torch.Tensor) -> torch.Tensor:
    """(N,3) radyan -> (N,3,3). numpy surumuyle ayni sozlesme (Blender XYZ)."""
    x, y, z = angles[:, 0], angles[:, 1], angles[:, 2]
    cx, sx, cy, sy, cz, sz = x.cos(), x.sin(), y.cos(), y.sin(), z.cos(), z.sin()
    zero = torch.zeros_like(x)
    rows = [
        torch.stack([cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz], dim=-1),
        torch.stack([cy * sz, cx * cz + sx * sy * sz, -cz * sx + cx * sy * sz], dim=-1),
        torch.stack([-sy, cy * sx, cx * cy], dim=-1),
    ]
    _ = zero
    return torch.stack(rows, dim=-2)


class TorchFaceModel(torch.nn.Module):
    """`facemodel.FaceModel`in turevlenebilir ikizi.

    Ayni matematik, ayni dogrulama (sifir kontrolde notr poz), ama gradyan
    akiyor ve GPU'ya tasinabiliyor.
    """

    def __init__(self, data: dict, dtype: torch.dtype = torch.float32):
        super().__init__()
        from .facemodel import FaceModel

        reference = FaceModel(data)  # aktif joint/seviye hesabini yeniden yazma
        self.levels = [torch.as_tensor(level, dtype=torch.long) for level in reference._levels]
        self.joint_count = reference.joint_count

        def register(name, array, dtype_override=None):
            self.register_buffer(
                name, torch.as_tensor(np.asarray(array), dtype=dtype_override or dtype), persistent=False
            )

        register("neutral", reference.neutral)
        register("skin_weights", reference.skin_weights)
        register("skin_joints", reference._skin_joints_local, torch.long)
        register("blend_deltas", reference.blend_deltas)
        register("blend_channels", reference.blend_channels, torch.long)
        register("active_joints", reference._active_joints, torch.long)
        register("local_parents", reference._local_parents, torch.long)
        register("joint_translation", reference.joint_neutral_translation[reference._active_joints])
        register("joint_rotation", reference.joint_neutral_rotation[reference._active_joints])
        register("bind_inverse", reference._bind_inverse)

    @classmethod
    def load(cls, path: Path | str, **kwargs) -> "TorchFaceModel":
        with np.load(Path(path), allow_pickle=False) as data:
            return cls({key: data[key] for key in data.files}, **kwargs)

    def forward(self, joint_outputs: torch.Tensor, blend_weights: torch.Tensor) -> torch.Tensor:
        """(B, 7830) + (B, 782) -> (B, vertex, 3)."""
        batch = joint_outputs.shape[0]

        # --- blendshape deltalari (bind uzayinda, skinning'den once)
        positions = self.neutral.unsqueeze(0).expand(batch, -1, -1)
        if len(self.blend_channels):
            selected = blend_weights[:, self.blend_channels]  # (B, T)
            positions = positions + torch.einsum("bt,tvc->bvc", selected, self.blend_deltas)

        # --- joint yerel donusumleri: notr + delta
        values = joint_outputs.view(batch, self.joint_count, JOINT_STRIDE)[:, self.active_joints]
        translation = self.joint_translation.unsqueeze(0) + values[..., 0:3]
        rotation_degrees = self.joint_rotation.unsqueeze(0) + values[..., 3:6]
        scale = 1.0 + values[..., 6:9]

        count = translation.shape[1]
        rotation = euler_xyz_to_matrix(
            torch.deg2rad(rotation_degrees).reshape(-1, 3)
        ).reshape(batch, count, 3, 3)

        local = torch.zeros(batch, count, 4, 4, dtype=positions.dtype, device=positions.device)
        local[..., :3, :3] = rotation * scale.unsqueeze(-2)
        local[..., :3, 3] = translation
        local[..., 3, 3] = 1.0

        # --- hiyerarsiyi seviye seviye zincirle (10 adim, joint sayisi kadar degil)
        world = local.clone()
        for depth, indices in enumerate(self.levels):
            if depth == 0 or not len(indices):
                continue
            parents = self.local_parents[indices]
            world = world.index_copy(1, indices, world[:, parents] @ local[:, indices])

        skinning = world @ self.bind_inverse.unsqueeze(0)

        # --- linear blend skinning
        homogeneous = torch.cat(
            [positions, torch.ones_like(positions[..., :1])], dim=-1
        )  # (B, V, 4)
        result = torch.zeros_like(positions)
        for slot in range(self.skin_joints.shape[1]):
            weight = self.skin_weights[:, slot]
            if not bool((weight > 1e-9).any()):
                continue
            matrices = skinning[:, self.skin_joints[:, slot]]  # (B, V, 4, 4)
            transformed = torch.einsum("bvij,bvj->bvi", matrices, homogeneous)
            result = result + weight.view(1, -1, 1) * transformed[..., :3]
        return result


def rigid_align(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """source'u target'a rijit + tek olcekle hizala (Procrustes).

    Kafa pozunu ve olcegi ayiklar; geriye sadece ifade farki kalir.
    Kamera ic parametrelerini bilmedigimiz icin 2D izdusum yerine bu.
    """
    source_center = source.mean(dim=0, keepdim=True)
    target_center = target.mean(dim=0, keepdim=True)
    a = source - source_center
    b = target - target_center

    covariance = a.T @ b
    u, s, vt = torch.linalg.svd(covariance)
    d = torch.sign(torch.det(vt.T @ u.T))
    correction = torch.diag(torch.tensor([1.0, 1.0, d], dtype=source.dtype, device=source.device))
    rotation = vt.T @ correction @ u.T

    scale = s.sum() / (a * a).sum().clamp_min(1e-12)
    return (scale * (rotation @ a.T).T) + target_center


def axis_angle_to_matrix(vector: torch.Tensor) -> torch.Tensor:
    """Rodrigues. Kucuk acilarda kararli olmasi icin sin/cos seri yerine
    clamp'li normalizasyon kullaniliyor."""
    angle = vector.norm().clamp_min(1e-8)
    axis = vector / angle
    x, y, z = axis[0], axis[1], axis[2]
    zero = torch.zeros((), dtype=vector.dtype, device=vector.device)
    cross = torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ]
    )
    eye = torch.eye(3, dtype=vector.dtype, device=vector.device)
    return eye + torch.sin(angle) * cross + (1 - torch.cos(angle)) * (cross @ cross)


class RigidPose(torch.nn.Module):
    """Ogrenilebilir rijit donusum (rotasyon + oteleme + tek olcek).

    Neden ogrenilebilir: pozu ONCE ayiklayip sonra ifadeyi cozmek dairesel.
    Poz ayiklamak icin "oynamayan" landmark'lar lazim, ama hangilerinin
    oynamadigini bilmek ifadeyi bilmeyi gerektiriyor. Cekimde en az oynayan
    landmark'lari secince Procrustes onlari notre ZORLUYOR ve gercek
    hareketlerini bastiriyor -- olculdu: gulumseme karesinde gozlemde
    "duran" 57 landmark'in 48'i tam da rijit sayilan kumedendi.

    Poz ile ifadeyi BIRLIKTE cozunce boyle bir taahhut gerekmiyor.
    """

    def __init__(self, dtype: torch.dtype, device: str):
        super().__init__()
        kwargs = {"dtype": dtype, "device": device}
        self.rotation = torch.nn.Parameter(torch.zeros(3, **kwargs))
        self.translation = torch.nn.Parameter(torch.zeros(3, **kwargs))
        self.log_scale = torch.nn.Parameter(torch.zeros((), **kwargs))

    @torch.no_grad()
    def initialize(self, source: torch.Tensor, target: torch.Tensor) -> None:
        """Procrustes ile baslat -- sifirdan baslamak yerel minimuma dusuruyor."""
        aligned = rigid_align(source, target)
        scale = (aligned - aligned.mean(0)).norm() / (source - source.mean(0)).norm().clamp_min(1e-9)
        self.log_scale.copy_(scale.log())
        self.translation.copy_(aligned.mean(0) - scale * source.mean(0))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        center = points.mean(dim=0, keepdim=True)
        rotated = (axis_angle_to_matrix(self.rotation) @ (points - center).T).T
        return self.log_scale.exp() * rotated + center + self.translation


class LandmarkSolver:
    """Gozlenen landmark'lardan GUI kontrollerini cozer."""

    def __init__(
        self,
        behavior_path: Path | str,
        face_model_path: Path | str,
        correspondence_path: Path | str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        weight_tolerance: float = 0.05,
    ):
        self.device = device
        self.dtype = dtype
        self.rig = TorchRigLogic.load(behavior_path, dtype=dtype).to(device)
        self.model = TorchFaceModel.load(face_model_path, dtype=dtype).to(device)

        payload = json.loads(Path(correspondence_path).read_text(encoding="utf-8"))
        with np.load(Path(face_model_path), allow_pickle=False) as data:
            subset = {int(v): i for i, v in enumerate(data["vertexIndices"])}

        rows, weights, landmarks, dropped = [], [], [], []
        for record in payload["correspondence"]:
            w = record["weights"]
            # Ucgen disina tasan karsiliklar konumu asiri ekstrapole eder;
            # olculdu: birkac tanesi -0.87'ye kadar gidiyor. Bunlari at.
            if min(w) < -weight_tolerance or max(w) > 1.0 + weight_tolerance:
                dropped.append(record["landmark"])
                continue
            if any(int(v) not in subset for v in record["vertices"]):
                dropped.append(record["landmark"])
                continue
            rows.append([subset[int(v)] for v in record["vertices"]])
            weights.append(w)
            landmarks.append(record["landmark"])

        self.landmark_ids = landmarks
        self.dropped = dropped
        self.rows = torch.tensor(rows, dtype=torch.long, device=device)
        self.weights = torch.tensor(weights, dtype=dtype, device=device)

    @property
    def landmark_count(self) -> int:
        return len(self.landmark_ids)

    def landmarks_from_gui(self, gui: torch.Tensor) -> torch.Tensor:
        """(B, 174) -> (B, landmark, 3)."""
        joint_outputs, blend_weights = self.rig(gui)
        vertices = self.model(joint_outputs, blend_weights)
        picked = vertices[:, self.rows]  # (B, L, 3, 3)
        return (picked * self.weights.unsqueeze(0).unsqueeze(-1)).sum(dim=2)

    def solve(
        self,
        observed: torch.Tensor,
        initial: torch.Tensor | None = None,
        iterations: int = 300,
        learning_rate: float = 0.08,
        regularization: float = 1e-4,
        sparsity: float = 5e-4,
        debias: bool = True,
        debias_threshold: float = 0.05,
        align: bool = True,
        joint_pose: bool = False,
        verbose: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """Gozlenen landmark'lardan (landmark, 3) GUI kontrollerini coz.

        `sparsity` (L1) neden sart: problem kotu kosullu, farkli kontrol
        kombinasyonlari benzer landmark dizilimi uretiyor. Sadece L2 ile
        cozdugumuzde 23 kontrol yanlislikla tetikleniyordu. Gercek ifadeler
        AZ SAYIDA kontrol kullanir; L1 bu onseli dayatiyor.

        Sentetik gidis-donus testinde olculdu (hedef 6 kontrol):

            L2      L1      RMS      kontrol hatasi   yanlis tetik
            1e-4    0       0.118mm  0.093            23
            1e-3    0       0.183mm  0.184             3
            1e-4    5e-4    0.112mm  0.066             1   <- varsayilan
            1e-4    3e-3    0.436mm  0.212             0

        Sadece L2'yi buyutmek her seyi bastiriyor; L1 yanlis tetikleri
        kirparken dogrulugu da iyilestiriyor.

        `debias` (iki asamali cozum): L1'in bilinen yan etkisi tahminleri
        sifira dogru buzmesidir -- yukaridaki testte kirpma 0.50 yerine
        0.435 cikiyordu. Standart caresi: L1 hangi kontrollerin AKTIF
        oldugunu bulsun, sonra sadece o kume uzerinde L1'siz yeniden coz.

            asama           ort. kontrol hatasi   RMS
            L1              0.066                 0.1118 mm
            + yanlilik gid. 0.002                 0.0060 mm
        """
        observed = observed.to(self.device, self.dtype)

        # poz ile ifadeyi birlikte coz (bkz. RigidPose). `align` ile birlikte
        # kullanilmaz: ikisi de ayni serbestligi tuketir, ust uste binince
        # gradyan birbirini yer.
        pose = None
        if joint_pose:
            pose = RigidPose(self.dtype, self.device)
            pose.initialize(observed, self.landmarks_from_gui(
                torch.zeros(1, self.rig.gui_count, dtype=self.dtype, device=self.device)
            )[0].detach())
            align = False

        gui = (
            torch.zeros(1, self.rig.gui_count, dtype=self.dtype, device=self.device)
            if initial is None
            else initial.clone().to(self.device, self.dtype).reshape(1, -1)
        )
        gui.requires_grad_(True)
        groups = [{"params": [gui], "lr": learning_rate}]
        if pose is not None:
            # poz parametreleri farkli olcekte; ayni lr ile kontroller
            # oturmadan poz kaciyor
            groups.append({"params": list(pose.parameters()), "lr": learning_rate * 0.25})
        optimizer = torch.optim.Adam(groups)
        # sonlara dogru adimi kucult: son iterasyonlarda salinim yerine oturma
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations)

        history = []
        for step in range(iterations):
            optimizer.zero_grad()
            predicted = self.landmarks_from_gui(gui)[0]
            if pose is not None:
                target = pose(observed)
            else:
                target = rigid_align(observed, predicted.detach()) if align else observed

            residual = ((predicted - target) ** 2).sum(dim=1)
            loss = (
                residual.mean()
                + regularization * (gui**2).sum()
                + sparsity * gui.abs().sum()
            )
            loss.backward()
            optimizer.step()
            schedule.step()

            with torch.no_grad():
                # GUI kontrolleri [-1, 1] araliginda tanimli
                gui.clamp_(-1.0, 1.0)

            history.append(loss.detach().item())
            if verbose and step % 25 == 0:
                print(
                    f"  {step:3d}  kayip={history[-1]:.6e}  "
                    f"rms={residual.detach().mean().sqrt().item()*10:.4f} mm"
                )

        result = gui.detach()[0]

        # --- 2. asama: yanlilik giderme
        if debias:
            mask = (result.abs() > debias_threshold).to(self.dtype)
            if bool(mask.any()):
                refined = result.clone().reshape(1, -1).requires_grad_(True)
                optimizer = torch.optim.Adam([refined], lr=learning_rate * 0.4)
                schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations // 2)
                for _ in range(iterations // 2):
                    optimizer.zero_grad()
                    masked = refined * mask
                    predicted = self.landmarks_from_gui(masked)[0]
                    if pose is not None:
                        target = pose(observed).detach()
                    else:
                        target = rigid_align(observed, predicted.detach()) if align else observed
                    loss = ((predicted - target) ** 2).sum(dim=1).mean() + 1e-5 * (masked**2).sum()
                    loss.backward()
                    optimizer.step()
                    schedule.step()
                    with torch.no_grad():
                        refined.clamp_(-1.0, 1.0)
                    history.append(loss.detach().item())
                result = (refined.detach() * mask)[0]

        with torch.no_grad():
            predicted = self.landmarks_from_gui(result.reshape(1, -1))[0]
            target = rigid_align(observed, predicted) if align else observed
            rms = float(((predicted - target) ** 2).sum(dim=1).mean().sqrt())

        return result, {
            "loss": history,
            "rms": rms,
            "rmsMillimeters": rms * 10,  # DNA birimi cm
            "iterations": iterations,
            "activeControls": int((result.abs() > 0.05).sum()),
        }
