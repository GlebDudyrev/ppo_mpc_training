from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

from stable_baselines3.common.callbacks import BaseCallback


@dataclass
class GradStat:
    count: int = 0
    finite_count: int = 0
    nonfinite_count: int = 0
    min_norm: float = math.inf
    max_norm: float = 0.0
    sum_norm: float = 0.0
    last_norm: float = 0.0

    def update(self, norm: float) -> None:
        self.count += 1
        self.last_norm = float(norm)
        if math.isfinite(norm):
            self.finite_count += 1
            self.min_norm = min(self.min_norm, float(norm))
            self.max_norm = max(self.max_norm, float(norm))
            self.sum_norm += float(norm)
        else:
            self.nonfinite_count += 1

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["mean_norm"] = self.sum_norm / self.finite_count if self.finite_count > 0 else 0.0
        if not math.isfinite(self.min_norm):
            payload["min_norm"] = 0.0
        return payload


class GradientMonitoringCallback(BaseCallback):
    """Collect gradient norm statistics for selected policy parameters during PPO training."""

    def __init__(
        self,
        save_path: str | Path,
        parameter_substrings: List[str] | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.save_path = Path(save_path)
        self.parameter_substrings = parameter_substrings or [
            "action_net.reference_head",
            "action_net.lqr.q_raw",
            "action_net.lqr.r_raw",
            "action_net.lqr.qf_raw",
        ]
        self._stats: Dict[str, GradStat] = {}
        self._hook_handles = []

    def _matches(self, name: str) -> bool:
        return any(substr in name for substr in self.parameter_substrings)

    def _on_training_start(self) -> None:
        self._stats = {}
        self._hook_handles = []

        for name, param in self.model.policy.named_parameters():
            if not self._matches(name):
                continue
            self._stats[name] = GradStat()

            def _make_hook(param_name: str):
                def _hook(grad):
                    norm = float(grad.norm().detach().cpu()) if grad is not None else float("nan")
                    self._stats[param_name].update(norm)
                return _hook

            handle = param.register_hook(_make_hook(name))
            self._hook_handles.append(handle)

        if self.verbose > 0:
            print("[gradient_monitor] hooks registered for:")
            for name in self._stats:
                print(f"  - {name}")

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "parameter_substrings": self.parameter_substrings,
            "stats": {name: stat.to_json() for name, stat in self._stats.items()},
        }
        with self.save_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=False)

        if self.verbose > 0:
            print(f"[gradient_monitor] saved gradient stats to {self.save_path}")
            for name, stat in payload["stats"].items():
                print(
                    f"  {name}: count={stat['count']} finite={stat['finite_count']} "
                    f"mean={stat['mean_norm']:.6g} min={stat['min_norm']:.6g} "
                    f"max={stat['max_norm']:.6g} nonfinite={stat['nonfinite_count']}"
                )
