import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.nn import functional as F


METHODS = [
    "contactcloak",
    "raw",
    "root_normalization",
    "fixed_root",
    "no_identity",
    "no_hard_constraint",
    "preprojection_privacy",
]
EVALUATION_NEEDS_GRAD = True


def rotations(axis_angle):
    x, y, z = axis_angle.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(
        *axis_angle.shape[:-1], 3, 3
    )
    return torch.matrix_exp(skew)


def forward_kinematics(pose, offsets, parents):
    root = pose[:3]
    root_rotation = rotations(pose[3:6])
    local = rotations(pose[6:].reshape(-1, 3))
    positions, orientation = [], []
    for j, parent in enumerate(parents):
        if parent < 0:
            positions.append(root + root_rotation @ offsets[j])
            orientation.append(root_rotation @ local[j])
        else:
            positions.append(positions[parent] + orientation[parent] @ offsets[j])
            orientation.append(orientation[parent] @ local[j])
    return torch.stack(positions)


class Model(nn.Module):
    def __init__(self, cfg, method):
        super().__init__()
        self.cfg, self.method = cfg, method
        joints = len(cfg["parents"])
        self.dim = 6 + 3 * joints
        self.proposal = nn.GRU(self.dim + 5 * joints, cfg["hidden"], batch_first=True)
        self.readout = nn.Linear(cfg["hidden"], self.dim)
        remote_width = self.dim + 5 * joints + 1
        self.adversary = nn.Sequential(
            nn.Linear(remote_width, cfg["hidden"]),
            nn.GELU(),
            nn.Linear(cfg["hidden"], cfg["identities"]),
        )
        self.register_buffer("offsets", torch.tensor(cfg["offsets"], dtype=torch.float32))

    def project(self, candidate, anchors, contact, offsets, original_root, previous):
        pose = candidate
        if previous:
            jerk_target = 3 * previous[-1]
            if len(previous) >= 2:
                jerk_target = jerk_target - 3 * previous[-2]
            if len(previous) >= 3:
                jerk_target = jerk_target + previous[-3]
            pose = (pose + self.cfg["jerk_weight"] * jerk_target) / (1 + self.cfg["jerk_weight"])
        for _ in range(self.cfg["projection_steps"]):
            if not pose.requires_grad:
                pose = pose.detach().requires_grad_(True)

            def residual(z):
                position = forward_kinematics(z, offsets, self.cfg["parents"])
                return ((position - anchors) * contact[:, None]).reshape(-1)

            error = residual(pose)
            jacobian = torch.autograd.functional.jacobian(
                residual, pose, create_graph=self.training
            )
            identity = torch.eye(self.dim, device=pose.device)
            system = jacobian.T @ jacobian + self.cfg["damping"] * identity
            delta = torch.linalg.solve(system, jacobian.T @ error)
            if self.method == "fixed_root":
                delta = torch.cat([torch.zeros_like(delta[:6]), delta[6:]])
            pose = pose - delta
            pose = torch.cat(
                [pose[:6], pose[6:].clamp(-self.cfg["joint_limit"], self.cfg["joint_limit"])]
            )
            if self.method == "fixed_root":
                pose = torch.cat([original_root, pose[6:]])
        return pose

    def forward(self, batch):
        x = batch["x"].float()
        anchors = batch["anchors"].float()
        contact = batch["contact"].float()
        bodies = []
        for seed in batch["session_seed"].detach().cpu().tolist():
            generator = torch.Generator().manual_seed(int(seed))
            bodies.append(0.8 + 0.4 * torch.rand(len(self.cfg["parents"]), generator=generator))
        beta = torch.stack(bodies).to(x.device)
        feature = torch.cat(
            [x, anchors.flatten(-2), contact, beta[:, None].expand(-1, x.shape[1], -1)], -1
        )
        hidden, _ = self.proposal(feature)
        candidate = self.readout(hidden)
        outputs, residuals, statuses, projected = [], [], [], []
        for b in range(len(x)):
            history, raw_projected, errors, valid = [], [], [], []
            offsets = self.offsets * beta[b, :, None]
            for t in range(x.shape[1]):
                proposal = candidate[b, t]
                if self.method == "raw":
                    pose = x[b, t]
                elif self.method == "root_normalization":
                    pose = torch.cat([torch.zeros_like(x[b, t, :6]), x[b, t, 6:]])
                elif self.method == "no_hard_constraint":
                    pose = proposal
                else:
                    pose = self.project(
                        proposal, anchors[b, t], contact[b, t], offsets, x[b, t, :6], history[-3:]
                    )
                position = forward_kinematics(pose, offsets, self.cfg["parents"])
                error = torch.linalg.vector_norm(position - anchors[b, t], dim=-1) * contact[b, t]
                ground = position[:, 1].min() >= -self.cfg["penetration_tolerance"]
                ok = (error.max() <= self.cfg["contact_tolerance"]) & ground
                fallback = torch.zeros_like(pose)
                if history:
                    previous_positions = forward_kinematics(
                        history[-1], offsets, self.cfg["parents"]
                    )
                    previous_error = (
                        torch.linalg.vector_norm(previous_positions - anchors[b, t], dim=-1)
                        * contact[b, t]
                    )
                    previous_ok = (previous_error.max() <= self.cfg["contact_tolerance"]) & (
                        previous_positions[:, 1].min() >= -self.cfg["penetration_tolerance"]
                    )
                    fallback = torch.where(previous_ok, history[-1], fallback)
                emitted = (
                    pose
                    if self.method in ("raw", "root_normalization", "no_hard_constraint")
                    else torch.where(ok, pose, fallback)
                )
                history.append(emitted)
                raw_projected.append(pose)
                emitted_position = forward_kinematics(emitted, offsets, self.cfg["parents"])
                emitted_error = (
                    torch.linalg.vector_norm(emitted_position - anchors[b, t], dim=-1)
                    * contact[b, t]
                )
                errors.append(torch.stack([error, emitted_error], -1))
                valid.append(ok)
            outputs.append(torch.stack(history))
            projected.append(torch.stack(raw_projected))
            residuals.append(torch.stack(errors))
            statuses.append(torch.stack(valid))
        pose = torch.stack(outputs)
        status = torch.stack(statuses)
        remote_pose = candidate if self.method == "preprojection_privacy" else pose
        remote = torch.cat(
            [
                remote_pose,
                anchors.flatten(-2),
                contact,
                beta[:, None].expand(-1, x.shape[1], -1),
                status[..., None].float(),
            ],
            -1,
        )
        identity_logits = self.adversary(remote).mean(1)
        return {
            "pose": pose,
            "projected": torch.stack(projected),
            "residual": torch.stack(residuals),
            "valid": status,
            "identity_logits": identity_logits,
            "remote": remote,
        }


def adversary_loss(model, batch, cfg):
    prediction = model(batch)
    logits = model.adversary(prediction["remote"].detach()).mean(1)
    return F.cross_entropy(logits, batch["y"].long())


def loss(model, batch, cfg):
    result = model(batch)
    contact = result["residual"][..., 0].square().mean() / cfg["contact_tolerance"] ** 2
    pose = result["projected"]
    smooth = torch.diff(pose, n=3, dim=1).square().mean() if pose.shape[1] >= 4 else pose.sum() * 0
    identity = F.cross_entropy(result["identity_logits"], batch["y"].long())
    return (
        contact
        + cfg["smooth_weight"] * smooth
        - (0 if model.method == "no_identity" else cfg["privacy_weight"]) * identity
    )


def predict(model, batch, cfg, calibration=None):
    return model(batch)


def metrics(prediction, data, cfg):
    contact = np.asarray(data["contact"]) > 0.5
    errors = prediction["residual"][..., 1][contact]
    valid = prediction["valid"].astype(bool)
    pose = prediction["pose"]
    return {
        "identity_top1": float(accuracy_score(data["y"], prediction["identity_logits"].argmax(-1))),
        "contact_error_cm": float(errors.mean() * 100) if errors.size else None,
        "contact_p95_cm": float(np.quantile(errors, 0.95) * 100) if errors.size else None,
        "task_success": float(valid.all(1).mean()),
        "valid_frame_fraction": float(valid.mean()),
        "solver_failure_fraction": float((~valid).mean()),
        "jerk_energy": float(np.square(np.diff(pose, n=3, axis=1)).mean())
        if pose.shape[1] >= 4
        else None,
    }


def inference(model, batch, cfg):
    return model(batch)


INFERENCE_NEEDS_GRAD = True
