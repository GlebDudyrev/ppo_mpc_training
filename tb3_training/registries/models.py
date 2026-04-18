from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict

from stable_baselines3 import PPO
from torch import nn

from tb3_training.policies import LQRActorCriticPolicy


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    build_fn: Callable[..., Any]
    load_fn: Callable[..., Any]
    default_hyperparams: dict = field(default_factory=dict)

    def build(self, **kwargs):
        params = dict(self.default_hyperparams)
        if "policy_kwargs" in params and "policy_kwargs" in kwargs:
            merged_policy_kwargs = dict(params["policy_kwargs"])
            merged_policy_kwargs.update(kwargs.pop("policy_kwargs"))
            params["policy_kwargs"] = merged_policy_kwargs
        params.update(kwargs)
        return self.build_fn(**params)

    def load(self, path: str, **kwargs):
        return self.load_fn(path, **kwargs)


_MODEL_REGISTRY: Dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec) -> None:
    _MODEL_REGISTRY[spec.name] = spec


def get_model_spec(name: str) -> ModelSpec:
    try:
        return _MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_MODEL_REGISTRY)) or "<empty>"
        raise KeyError(f"Unknown model '{name}'. Available: {available}") from exc


def list_models() -> list[str]:
    return sorted(_MODEL_REGISTRY)


def build_ppo_model(**kwargs):
    return PPO(**kwargs)


def load_ppo_model(path: str, **kwargs):
    return PPO.load(path, **kwargs)


register_model(
    ModelSpec(
        name="pure_rl",
        description="Standard PPO with MlpPolicy directly outputting (v, w).",
        build_fn=build_ppo_model,
        load_fn=load_ppo_model,
        default_hyperparams={"policy": "MlpPolicy"},
    )
)

register_model(
    ModelSpec(
        name="static_lqr",
        description=(
            "PPO with the standard MLP actor backbone and a differentiable "
            "trainable-static LQR action_net replacing the final linear action head."
        ),
        build_fn=build_ppo_model,
        load_fn=load_ppo_model,
        default_hyperparams={
            "policy": LQRActorCriticPolicy,
            "policy_kwargs": {
                "net_arch": {"pi": [64, 64], "vf": [64, 64]},
                "activation_fn": nn.Tanh,
                "lqr_horizon": 5,
                "lqr_dt": 0.10,
                "x_ref_max": 1.0,
                "y_ref_max": 0.8,
                "theta_ref_max": 1.5707963267948966,
                "max_linear_velocity": 0.22,
                "max_angular_velocity": 2.8,
                "q_init": (2.0, 6.0, 4.0),
                "r_init": (0.5, 0.8),
                "qf_init": (4.0, 10.0, 6.0),
            },
        },
    )
)
