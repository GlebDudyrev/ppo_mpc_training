from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict

from stable_baselines3 import PPO


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    build_fn: Callable[..., Any]
    load_fn: Callable[..., Any]
    default_hyperparams: dict = field(default_factory=dict)

    def build(self, **kwargs):
        params = dict(self.default_hyperparams)
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


def build_pure_rl_model(**kwargs):
    return PPO(**kwargs)


def load_pure_rl_model(path: str, **kwargs):
    return PPO.load(path, **kwargs)


register_model(
    ModelSpec(
        name="pure_rl",
        description="Standard PPO with MlpPolicy directly outputting (v, w).",
        build_fn=build_pure_rl_model,
        load_fn=load_pure_rl_model,
        default_hyperparams={"policy": "MlpPolicy"},
    )
)
