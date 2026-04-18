from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

import numpy as np


EPISODE_COLUMNS = [
    "episode",
    "return",
    "steps",
    "success",
    "collision",
    "timeout",
    "initial_distance_to_goal",
    "final_distance_to_goal",
    "path_length",
    "path_efficiency",
    "episode_min_lidar_distance",
    "avg_abs_linear_speed",
    "avg_abs_angular_speed",
    "mean_action_smoothness",
    "action_smoothness_sum",
    "v_saturation_rate",
    "omega_saturation_rate",
    "final_robot_x",
    "final_robot_y",
    "final_robot_yaw",
    "target_x",
    "target_y",
    "scene_name",
    "world_name",
]


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return False


def _mean_finite(values: Iterable[float], default: float = 0.0) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return mean(finite) if finite else default


@dataclass
class EpisodeMetricsRecorder:
    episode_index: int
    initial_info: dict[str, Any] | None = None
    rewards: list[float] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    positions: list[tuple[float, float]] = field(default_factory=list)
    yaws: list[float] = field(default_factory=list)
    min_lidar_values: list[float] = field(default_factory=list)
    linear_speeds: list[float] = field(default_factory=list)
    angular_speeds: list[float] = field(default_factory=list)
    last_info: dict[str, Any] = field(default_factory=dict)
    terminated: bool = False
    truncated: bool = False
    max_linear_velocity: float = 0.22
    max_angular_velocity: float = 2.8
    saturation_fraction: float = 0.98

    def start(self, initial_info: dict[str, Any]) -> None:
        self.initial_info = dict(initial_info)
        self.last_info = dict(initial_info)
        self._record_info(initial_info)

    def record_step(
        self,
        *,
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        action_array = np.asarray(action, dtype=np.float64).reshape(-1)
        if action_array.size >= 2:
            self.actions.append(action_array[:2].copy())
        else:
            padded = np.zeros(2, dtype=np.float64)
            padded[: action_array.size] = action_array
            self.actions.append(padded)

        self.rewards.append(float(reward))
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        self.last_info = dict(info)
        self._record_info(info)

    def _record_info(self, info: dict[str, Any]) -> None:
        robot_x = _safe_float(info.get("robot_x"))
        robot_y = _safe_float(info.get("robot_y"))
        if math.isfinite(robot_x) and math.isfinite(robot_y):
            self.positions.append((robot_x, robot_y))

        robot_yaw = _safe_float(info.get("robot_yaw"))
        if math.isfinite(robot_yaw):
            self.yaws.append(robot_yaw)

        min_lidar = _safe_float(info.get("min_lidar_distance"))
        if math.isfinite(min_lidar):
            self.min_lidar_values.append(min_lidar)

        linear_velocity = _safe_float(info.get("linear_velocity"))
        if math.isfinite(linear_velocity):
            self.linear_speeds.append(abs(linear_velocity))

        angular_velocity = _safe_float(info.get("angular_velocity"))
        if math.isfinite(angular_velocity):
            self.angular_speeds.append(abs(angular_velocity))

    def _path_length(self) -> float:
        if len(self.positions) < 2:
            return 0.0
        total = 0.0
        for (x0, y0), (x1, y1) in zip(self.positions[:-1], self.positions[1:]):
            total += math.hypot(x1 - x0, y1 - y0)
        return float(total)

    def _action_smoothness(self) -> tuple[float, float]:
        if not self.actions:
            return 0.0, 0.0
        previous = np.zeros_like(self.actions[0])
        deltas = []
        for action in self.actions:
            delta = action - previous
            deltas.append(float(np.dot(delta, delta)))
            previous = action
        smoothness_sum = float(sum(deltas))
        smoothness_mean = float(smoothness_sum / len(deltas)) if deltas else 0.0
        return smoothness_mean, smoothness_sum

    def _saturation_rates(self) -> tuple[float, float]:
        if not self.actions:
            return 0.0, 0.0
        v_threshold = self.saturation_fraction * self.max_linear_velocity
        omega_threshold = self.saturation_fraction * self.max_angular_velocity
        v_count = sum(abs(float(action[0])) >= v_threshold for action in self.actions)
        omega_count = sum(abs(float(action[1])) >= omega_threshold for action in self.actions)
        total = len(self.actions)
        return float(v_count / total), float(omega_count / total)

    def finish(self) -> dict[str, Any]:
        info = self.last_info or {}
        initial_info = self.initial_info or {}
        steps = len(self.rewards)
        episode_return = float(sum(self.rewards))

        success = _safe_bool(info.get("success", False))
        collision = _safe_bool(info.get("collision", False))
        timeout = _safe_bool(info.get("timeout", self.truncated and not success and not collision))

        initial_distance = _safe_float(
            initial_info.get("distance_to_goal", info.get("initial_distance_to_goal"))
        )
        final_distance = _safe_float(info.get("distance_to_goal"))
        path_length = self._path_length()
        path_efficiency = 0.0
        if path_length > 1e-9 and math.isfinite(initial_distance):
            path_efficiency = float(min(initial_distance / path_length, 1.0))

        min_lidar = min(self.min_lidar_values) if self.min_lidar_values else float("nan")
        mean_action_smoothness, action_smoothness_sum = self._action_smoothness()
        v_saturation_rate, omega_saturation_rate = self._saturation_rates()

        final_x = _safe_float(info.get("robot_x"))
        final_y = _safe_float(info.get("robot_y"))
        final_yaw = _safe_float(info.get("robot_yaw"))

        return {
            "episode": int(self.episode_index),
            "return": episode_return,
            "steps": int(steps),
            "success": success,
            "collision": collision,
            "timeout": timeout,
            "initial_distance_to_goal": initial_distance,
            "final_distance_to_goal": final_distance,
            "path_length": path_length,
            "path_efficiency": path_efficiency,
            "episode_min_lidar_distance": float(min_lidar),
            "avg_abs_linear_speed": _mean_finite(self.linear_speeds),
            "avg_abs_angular_speed": _mean_finite(self.angular_speeds),
            "mean_action_smoothness": mean_action_smoothness,
            "action_smoothness_sum": action_smoothness_sum,
            "v_saturation_rate": v_saturation_rate,
            "omega_saturation_rate": omega_saturation_rate,
            "final_robot_x": final_x,
            "final_robot_y": final_y,
            "final_robot_yaw": final_yaw,
            "target_x": _safe_float(info.get("target_x")),
            "target_y": _safe_float(info.get("target_y")),
            "scene_name": info.get("scene_name", initial_info.get("scene_name")),
            "world_name": info.get("world_name", initial_info.get("world_name")),
        }


@dataclass
class EvaluationBenchmark:
    episodes: list[dict[str, Any]] = field(default_factory=list)

    def add_episode(self, metrics: dict[str, Any]) -> None:
        self.episodes.append(dict(metrics))

    def summary(self) -> dict[str, Any]:
        returns = [float(row["return"]) for row in self.episodes]
        steps = [int(row["steps"]) for row in self.episodes]
        successes = [1.0 if row.get("success") else 0.0 for row in self.episodes]
        collisions = [1.0 if row.get("collision") else 0.0 for row in self.episodes]
        timeouts = [1.0 if row.get("timeout") else 0.0 for row in self.episodes]
        successful_path_eff = [
            float(row["path_efficiency"])
            for row in self.episodes
            if row.get("success") and math.isfinite(float(row["path_efficiency"]))
        ]

        return {
            "episodes": len(self.episodes),
            "mean_return": mean(returns) if returns else 0.0,
            "std_return": pstdev(returns) if len(returns) > 1 else 0.0,
            "mean_steps": mean(steps) if steps else 0.0,
            "success_rate": mean(successes) if successes else 0.0,
            "collision_rate": mean(collisions) if collisions else 0.0,
            "timeout_rate": mean(timeouts) if timeouts else 0.0,
            "mean_final_distance": self._mean_key("final_distance_to_goal"),
            "mean_path_length": self._mean_key("path_length"),
            "mean_path_efficiency": self._mean_key("path_efficiency"),
            "mean_success_path_efficiency": mean(successful_path_eff) if successful_path_eff else 0.0,
            "mean_episode_min_lidar_distance": self._mean_key("episode_min_lidar_distance"),
            "mean_avg_abs_linear_speed": self._mean_key("avg_abs_linear_speed"),
            "mean_avg_abs_angular_speed": self._mean_key("avg_abs_angular_speed"),
            "mean_action_smoothness": self._mean_key("mean_action_smoothness"),
            "mean_v_saturation_rate": self._mean_key("v_saturation_rate"),
            "mean_omega_saturation_rate": self._mean_key("omega_saturation_rate"),
        }

    def _mean_key(self, key: str) -> float:
        return _mean_finite([_safe_float(row.get(key)) for row in self.episodes])

    def save(self, run_dir: Path, *, output_json: str | None = None) -> tuple[Path, Path]:
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = Path(output_json).expanduser().resolve() if output_json else run_dir / "eval_summary.json"
        episodes_path = run_dir / "episodes.csv"

        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as fp:
            json.dump(self.summary(), fp, indent=2, ensure_ascii=False)

        with episodes_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=EPISODE_COLUMNS)
            writer.writeheader()
            for row in self.episodes:
                writer.writerow({key: row.get(key, "") for key in EPISODE_COLUMNS})

        return summary_path, episodes_path
