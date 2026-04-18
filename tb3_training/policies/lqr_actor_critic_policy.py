from __future__ import annotations

import math
from typing import Any, Optional

import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

from tb3_training.layers import StaticDifferentiableLQR


class LQRActionNet(nn.Module):
    """SB3 action_net replacement: latent_pi -> reference -> LQR -> mean action."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        *,
        horizon: int = 5,
        dt: float = 0.10,
        x_ref_max: float = 1.0,
        y_ref_max: float = 0.8,
        theta_ref_max: float = math.pi / 2.0,
        max_linear_velocity: float = 0.22,
        max_angular_velocity: float = 2.8,
        q_init: tuple[float, float, float] = (2.0, 6.0, 4.0),
        r_init: tuple[float, float] = (0.5, 0.8),
        qf_init: tuple[float, float, float] = (4.0, 10.0, 6.0),
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        if action_dim != 2:
            raise ValueError(f"LQRActionNet supports only 2D actions [v, omega], got action_dim={action_dim}")

        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.x_ref_max = float(x_ref_max)
        self.y_ref_max = float(y_ref_max)
        self.theta_ref_max = float(theta_ref_max)
        self.max_linear_velocity = float(max_linear_velocity)
        self.max_angular_velocity = float(max_angular_velocity)

        self.reference_head = nn.Linear(self.latent_dim, 5)
        self.lqr = StaticDifferentiableLQR(
            horizon=horizon,
            dt=dt,
            max_linear_velocity=max_linear_velocity,
            max_angular_velocity=max_angular_velocity,
            q_init=q_init,
            r_init=r_init,
            qf_init=qf_init,
            eps=eps,
            clamp_output=True,
        )

    def scale_reference(self, raw_reference: torch.Tensor) -> torch.Tensor:
        if raw_reference.ndim != 2 or raw_reference.shape[-1] != 5:
            raise ValueError(f"Expected raw_reference shape (batch, 5), got {tuple(raw_reference.shape)}")

        x_ref = self.x_ref_max * torch.tanh(raw_reference[:, 0])
        y_ref = self.y_ref_max * torch.tanh(raw_reference[:, 1])
        theta_ref = self.theta_ref_max * torch.tanh(raw_reference[:, 2])
        v_ref = self.max_linear_velocity * torch.tanh(raw_reference[:, 3])
        omega_ref = self.max_angular_velocity * torch.tanh(raw_reference[:, 4])
        return torch.stack([x_ref, y_ref, theta_ref, v_ref, omega_ref], dim=-1)

    def forward(self, latent_pi: torch.Tensor) -> torch.Tensor:
        raw_reference = self.reference_head(latent_pi)
        reference = self.scale_reference(raw_reference)
        return self.lqr(reference)

    def get_lqr_costs(self) -> dict[str, list[float]]:
        values = self.lqr.debug_values()
        return {"q": values.q, "r": values.r, "qf": values.qf}


class LQRActorCriticPolicy(ActorCriticPolicy):
    """ActorCriticPolicy with differentiable LQR as the actor action_net.

    The standard SB3 actor MLP still produces `latent_pi`. The usual linear
    `action_net` is replaced by:

        latent_pi -> reference_head -> StaticDifferentiableLQR -> mean action.

    The critic branch remains standard.
    """

    def __init__(
        self,
        *args: Any,
        lqr_horizon: int = 5,
        lqr_dt: float = 0.10,
        x_ref_max: float = 1.0,
        y_ref_max: float = 0.8,
        theta_ref_max: float = math.pi / 2.0,
        max_linear_velocity: float = 0.22,
        max_angular_velocity: float = 2.8,
        q_init: tuple[float, float, float] = (2.0, 6.0, 4.0),
        r_init: tuple[float, float] = (0.5, 0.8),
        qf_init: tuple[float, float, float] = (4.0, 10.0, 6.0),
        lqr_eps: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        self.lqr_horizon = int(lqr_horizon)
        self.lqr_dt = float(lqr_dt)
        self.x_ref_max = float(x_ref_max)
        self.y_ref_max = float(y_ref_max)
        self.theta_ref_max = float(theta_ref_max)
        self.max_linear_velocity = float(max_linear_velocity)
        self.max_angular_velocity = float(max_angular_velocity)
        self.q_init = q_init
        self.r_init = r_init
        self.qf_init = qf_init
        self.lqr_eps = float(lqr_eps)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:  # type: ignore[override]
        # Let SB3 build the standard extractor, value_net, distribution parameters, etc.
        super()._build(lr_schedule)

        if not isinstance(self.action_space, spaces.Box):
            raise ValueError("LQRActorCriticPolicy requires a continuous Box action space")
        action_dim = int(self.action_space.shape[0])

        # Replace the default linear action head with the differentiable LQR actor head.
        self.action_net = LQRActionNet(
            latent_dim=self.mlp_extractor.latent_dim_pi,
            action_dim=action_dim,
            horizon=self.lqr_horizon,
            dt=self.lqr_dt,
            x_ref_max=self.x_ref_max,
            y_ref_max=self.y_ref_max,
            theta_ref_max=self.theta_ref_max,
            max_linear_velocity=self.max_linear_velocity,
            max_angular_velocity=self.max_angular_velocity,
            q_init=self.q_init,
            r_init=self.r_init,
            qf_init=self.qf_init,
            eps=self.lqr_eps,
        ).to(self.device)

        # Important: `super()._build()` already created the optimizer before we
        # replaced `action_net`. Recreate it so reference_head and LQR parameters
        # are included in PPO optimization.
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def get_lqr_costs(self) -> Optional[dict[str, list[float]]]:
        if hasattr(self.action_net, "get_lqr_costs"):
            return self.action_net.get_lqr_costs()
        return None
