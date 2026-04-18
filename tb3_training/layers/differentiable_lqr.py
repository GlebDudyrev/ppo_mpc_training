from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LQRDebugValues:
    q: list[float]
    r: list[float]
    qf: list[float]


class StaticDifferentiableLQR(nn.Module):
    """Finite-horizon differentiable LQR layer with trainable global costs.

    The layer receives a local reference
        [x_ref, y_ref, theta_ref, v_ref, omega_ref]
    and returns the mean action [v, omega].

    Q, R and Qf are global trainable parameters of the layer. They do not
    depend on the observation, but they are optimized together with the PPO
    policy because this module lives inside `policy.action_net`.
    """

    def __init__(
        self,
        *,
        horizon: int = 5,
        dt: float = 0.10,
        max_linear_velocity: float = 0.22,
        max_angular_velocity: float = 2.8,
        q_init: tuple[float, float, float] = (2.0, 6.0, 4.0),
        r_init: tuple[float, float] = (0.5, 0.8),
        qf_init: tuple[float, float, float] = (4.0, 10.0, 6.0),
        eps: float = 1e-4,
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        if horizon < 1:
            raise ValueError("LQR horizon must be >= 1")
        if dt <= 0:
            raise ValueError("LQR dt must be positive")

        self.horizon = int(horizon)
        self.dt = float(dt)
        self.max_linear_velocity = float(max_linear_velocity)
        self.max_angular_velocity = float(max_angular_velocity)
        self.eps = float(eps)
        self.clamp_output = bool(clamp_output)

        self.q_raw = nn.Parameter(self._positive_inverse_softplus(torch.as_tensor(q_init, dtype=torch.float32)))
        self.r_raw = nn.Parameter(self._positive_inverse_softplus(torch.as_tensor(r_init, dtype=torch.float32)))
        self.qf_raw = nn.Parameter(self._positive_inverse_softplus(torch.as_tensor(qf_init, dtype=torch.float32)))

    def _positive_inverse_softplus(self, values: torch.Tensor) -> torch.Tensor:
        values = torch.clamp(values - self.eps, min=1e-6)
        return torch.log(torch.expm1(values))

    def positive_costs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = F.softplus(self.q_raw) + self.eps
        r = F.softplus(self.r_raw) + self.eps
        qf = F.softplus(self.qf_raw) + self.eps
        return q, r, qf

    def debug_values(self) -> LQRDebugValues:
        with torch.no_grad():
            q, r, qf = self.positive_costs()
            return LQRDebugValues(q=q.detach().cpu().tolist(), r=r.detach().cpu().tolist(), qf=qf.detach().cpu().tolist())

    def _build_linearized_matrices(self, v_ref: torch.Tensor, omega_ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = v_ref.shape[0]
        dtype = v_ref.dtype
        device = v_ref.device
        dt = torch.as_tensor(self.dt, dtype=dtype, device=device)

        A = torch.zeros(batch_size, 3, 3, dtype=dtype, device=device)
        A[:, 0, 0] = 1.0
        A[:, 1, 1] = 1.0
        A[:, 2, 2] = 1.0
        A[:, 0, 1] = dt * omega_ref
        A[:, 1, 0] = -dt * omega_ref
        A[:, 1, 2] = dt * v_ref

        B = torch.zeros(batch_size, 3, 2, dtype=dtype, device=device)
        B[:, 0, 0] = -dt
        B[:, 2, 1] = -dt
        return A, B

    def forward(self, reference: torch.Tensor) -> torch.Tensor:
        if reference.ndim != 2 or reference.shape[-1] != 5:
            raise ValueError(f"Expected reference shape (batch, 5), got {tuple(reference.shape)}")

        error = reference[:, 0:3]
        u_ref = reference[:, 3:5]
        v_ref = u_ref[:, 0]
        omega_ref = u_ref[:, 1]

        A, B = self._build_linearized_matrices(v_ref, omega_ref)
        batch_size = reference.shape[0]
        dtype = reference.dtype
        device = reference.device

        q, r, qf = self.positive_costs()
        q = q.to(dtype=dtype, device=device)
        r = r.to(dtype=dtype, device=device)
        qf = qf.to(dtype=dtype, device=device)

        Q = torch.diag(q).expand(batch_size, 3, 3)
        R = torch.diag(r).expand(batch_size, 2, 2)
        P = torch.diag(qf).expand(batch_size, 3, 3)

        K0 = None
        eye_u = torch.eye(2, dtype=dtype, device=device).expand(batch_size, 2, 2)

        for step in reversed(range(self.horizon)):
            BtP = B.transpose(1, 2).bmm(P)
            S = R + BtP.bmm(B) + self.eps * eye_u
            rhs = BtP.bmm(A)
            K = torch.linalg.solve(S, rhs)
            P = Q + A.transpose(1, 2).bmm(P).bmm(A) - A.transpose(1, 2).bmm(P).bmm(B).bmm(K)
            if step == 0:
                K0 = K

        if K0 is None:  # pragma: no cover
            raise RuntimeError("Riccati recursion did not produce K0")

        delta_u = -K0.bmm(error.unsqueeze(-1)).squeeze(-1)
        mean_action = u_ref + delta_u

        if self.clamp_output:
            v = torch.clamp(mean_action[:, 0], -self.max_linear_velocity, self.max_linear_velocity)
            omega = torch.clamp(mean_action[:, 1], -self.max_angular_velocity, self.max_angular_velocity)
            mean_action = torch.stack([v, omega], dim=-1)

        return mean_action
