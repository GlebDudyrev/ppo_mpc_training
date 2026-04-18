from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import rclpy

from tb3_training.benchmark import EpisodeMetricsRecorder, EvaluationBenchmark
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


def save_eval_config(run_dir: Path, config: EvalRunConfig, train_run_dir: Path) -> None:
    payload = {
        **asdict(config),
        "phase": "eval",
        "train_run_dir": str(train_run_dir),
        "artifacts": {
            "summary": "eval_summary.json",
            "episodes": "episodes.csv",
        },
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
        save_eval_config(eval_run_dir, config, train_run_dir)

        env_spec = get_env_spec(config.env_name)
        model_spec = get_model_spec(config.model_name)
        env = env_spec.make(mode="eval")
        env.reset(seed=config.seed)
        model = model_spec.load(str(model_path), device=config.device)

        benchmark = EvaluationBenchmark()
        for episode_idx in range(config.episodes):
            obs, info = env.reset(seed=config.seed + episode_idx)
            recorder = EpisodeMetricsRecorder(episode_index=episode_idx)
            recorder.start(info)

            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=config.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                recorder.record_step(
                    action=action,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )

            metrics = recorder.finish()
            benchmark.add_episode(metrics)
            print(
                "[tb3_eval] "
                f"episode={episode_idx + 1}/{config.episodes} "
                f"steps={metrics['steps']} return={metrics['return']:.3f} "
                f"final_distance={metrics['final_distance_to_goal']:.3f} "
                f"collision={metrics['collision']} success={metrics['success']} "
                f"v_sat={metrics['v_saturation_rate']:.3f} w_sat={metrics['omega_saturation_rate']:.3f}"
            )

        summary_path, episodes_path = benchmark.save(eval_run_dir, output_json=config.output_json)
        summary = benchmark.summary()
        print(
            "[tb3_eval] summary: "
            f"mean_return={summary['mean_return']:.3f}, "
            f"success_rate={summary['success_rate']:.3f}, "
            f"collision_rate={summary['collision_rate']:.3f}, "
            f"timeout_rate={summary['timeout_rate']:.3f}, "
            f"mean_steps={summary['mean_steps']:.2f}, "
            f"mean_final_distance={summary['mean_final_distance']:.3f}, "
            f"mean_v_sat={summary['mean_v_saturation_rate']:.3f}, "
            f"mean_w_sat={summary['mean_omega_saturation_rate']:.3f}"
        )
        print(f"[tb3_eval] saved summary to {summary_path}")
        print(f"[tb3_eval] saved episode details to {episodes_path}")
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
