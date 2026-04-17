from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Optional

import numpy as np
import rclpy

from tb3_training.experiments import ExperimentNamespace, make_version_dir, resolve_version_dir
from tb3_training.registries import get_env_spec, get_model_spec


@dataclass
class EvalRunConfig:
    model_name: str = "pure_rl"
    env_name: str = "simple_env"
    train_version: Optional[int] = None
    episodes: int = 10
    seed: int = 42
    deterministic: bool = True
    experiments_root: Optional[str] = None
    output_json: Optional[str] = None
    device: str = "auto"


def episode_metrics(info: dict[str, Any], episode_return: float, steps: int) -> dict[str, Any]:
    return {
        "return": float(episode_return),
        "steps": int(steps),
        "distance_to_goal": float(info.get("distance_to_goal", np.nan)),
        "min_lidar_distance": float(info.get("min_lidar_distance", np.nan)),
        "collision": bool(info.get("collision", False)),
        "success": bool(info.get("success", False)),
        "scene_name": info.get("scene_name"),
        "world_name": info.get("world_name"),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [r["return"] for r in results]
    steps = [r["steps"] for r in results]
    successes = [1.0 if r["success"] else 0.0 for r in results]
    collisions = [1.0 if r["collision"] else 0.0 for r in results]
    distances = [r["distance_to_goal"] for r in results]
    mins = [r["min_lidar_distance"] for r in results]
    return {
        "episodes": len(results),
        "mean_return": mean(returns) if returns else 0.0,
        "std_return": pstdev(returns) if len(returns) > 1 else 0.0,
        "mean_steps": mean(steps) if steps else 0.0,
        "success_rate": mean(successes) if successes else 0.0,
        "collision_rate": mean(collisions) if collisions else 0.0,
        "mean_final_distance": mean(distances) if distances else 0.0,
        "mean_min_lidar_distance": mean(mins) if mins else 0.0,
        "episodes_detail": results,
    }


def save_eval_config(
    run_dir: Path,
    config: EvalRunConfig,
    train_run_dir: Path,
    env_metadata: dict | None = None,
) -> None:
    payload = {
        **asdict(config),
        "phase": "eval",
        "train_run_dir": str(train_run_dir),
        "env_metadata": env_metadata or {},
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)


def run_eval(config: EvalRunConfig) -> int:
    env = None
    rclpy_inited = False

    try:
        if not rclpy.ok():
            rclpy.init(args=None)
            rclpy_inited = True

        train_namespace = ExperimentNamespace(
            model_name=config.model_name,
            env_name=config.env_name,
            phase="train",
        )
        train_run_dir = resolve_version_dir(
            train_namespace,
            root=config.experiments_root,
            version=config.train_version,
        )
        model_path = train_run_dir / "final_model" / "model"

        eval_namespace = ExperimentNamespace(
            model_name=config.model_name,
            env_name=config.env_name,
            phase="eval",
        )
        eval_run_dir = make_version_dir(eval_namespace, root=config.experiments_root)

        env_spec = get_env_spec(config.env_name)
        model_spec = get_model_spec(config.model_name)
        save_eval_config(eval_run_dir, config, train_run_dir, env_spec.metadata())

        env = env_spec.make(mode="eval")
        env.reset(seed=config.seed)
        model = model_spec.load(str(model_path), device=config.device)

        all_results: list[dict[str, Any]] = []
        for episode_idx in range(config.episodes):
            obs, info = env.reset(seed=config.seed + episode_idx)
            terminated = False
            truncated = False
            episode_return = 0.0
            steps = 0
            last_info = info

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=config.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                steps += 1
                last_info = info

            metrics = episode_metrics(last_info, episode_return, steps)
            all_results.append(metrics)
            print(
                "[tb3_eval] "
                f"episode={episode_idx + 1}/{config.episodes} "
                f"steps={metrics['steps']} return={metrics['return']:.3f} "
                f"distance_to_goal={metrics['distance_to_goal']:.3f} "
                f"collision={metrics['collision']} success={metrics['success']}"
            )

        summary = summarize(all_results)
        summary_path = Path(config.output_json) if config.output_json else eval_run_dir / "eval_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2, ensure_ascii=False)

        print(
            "[tb3_eval] summary: "
            f"mean_return={summary['mean_return']:.3f}, "
            f"success_rate={summary['success_rate']:.3f}, "
            f"collision_rate={summary['collision_rate']:.3f}, "
            f"mean_steps={summary['mean_steps']:.2f}, "
            f"mean_final_distance={summary['mean_final_distance']:.3f}, "
            f"mean_min_lidar_distance={summary['mean_min_lidar_distance']:.3f}"
        )
        print(f"[tb3_eval] saved summary to {summary_path}")
        return 0

    except KeyboardInterrupt:
        print("[tb3_eval] interrupted by user")
        return 130
    except Exception as exc:
        print(f"[tb3_eval] evaluation failed: {exc}")
        return 1
    finally:
        if env is not None:
            env.close()
        if rclpy_inited and rclpy.ok():
            rclpy.shutdown()
