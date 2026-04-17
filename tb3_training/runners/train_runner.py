from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import rclpy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from tb3_training.experiments import ExperimentNamespace, make_version_dir
from tb3_training.registries import get_env_spec, get_model_spec


@dataclass
class TrainRunConfig:
    model_name: str = "pure_rl"
    env_name: str = "simple_env"
    total_timesteps: int = 200_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    seed: int = 42
    checkpoint_freq: int = 10_000
    experiments_root: Optional[str] = None
    progress_bar: bool = True
    device: str = "auto"


def save_train_config(run_dir: Path, config: TrainRunConfig, env_metadata: dict | None = None) -> None:
    payload = {
        **asdict(config),
        "phase": "train",
        "env_metadata": env_metadata or {},
        "artifacts": {
            "tensorboard": "tensorboard",
            "checkpoints": "checkpoints",
            "final_model": "final_model/model",
        },
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)


def run_train(config: TrainRunConfig) -> int:
    train_env = None
    model = None
    run_dir: Optional[Path] = None
    rclpy_inited = False

    try:
        if not rclpy.ok():
            rclpy.init(args=None)
            rclpy_inited = True

        set_random_seed(config.seed)

        namespace = ExperimentNamespace(
            model_name=config.model_name,
            env_name=config.env_name,
            phase="train",
        )
        run_dir = make_version_dir(namespace, root=config.experiments_root)
        tensorboard_dir = run_dir / "tensorboard"
        checkpoints_dir = run_dir / "checkpoints"
        final_model_dir = run_dir / "final_model"

        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        final_model_dir.mkdir(parents=True, exist_ok=True)

        env_spec = get_env_spec(config.env_name)
        model_spec = get_model_spec(config.model_name)
        save_train_config(run_dir, config, env_spec.metadata())

        train_env_raw = env_spec.make(mode="train")
        train_env_raw.reset(seed=config.seed)
        train_env = Monitor(train_env_raw)

        checkpoint_callback = CheckpointCallback(
            save_freq=config.checkpoint_freq,
            save_path=str(checkpoints_dir),
            name_prefix=f"{config.model_name}_{config.env_name}",
        )

        model = model_spec.build(
            env=train_env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            seed=config.seed,
            verbose=1,
            tensorboard_log=str(tensorboard_dir),
            device=config.device,
        )

        print(f"[tb3_train] run_dir={run_dir}")
        print(f"[tb3_train] model={config.model_name} env={config.env_name} world={env_spec.world_name}")
        print(f"[tb3_train] total_timesteps={config.total_timesteps}, seed={config.seed}")

        model.learn(
            total_timesteps=config.total_timesteps,
            callback=checkpoint_callback,
            progress_bar=config.progress_bar,
        )

        final_model_path = final_model_dir / "model"
        model.save(str(final_model_path))
        print(f"[tb3_train] training completed successfully, saved to {final_model_path}")
        return 0

    except KeyboardInterrupt:
        print("[tb3_train] interrupted by user, saving current model state")
        if model is not None and run_dir is not None:
            interrupted_path = run_dir / "interrupted_model"
            model.save(str(interrupted_path))
            print(f"[tb3_train] interrupted model saved to {interrupted_path}")
        return 130
    except Exception as exc:
        print(f"[tb3_train] training failed: {exc}")
        return 1
    finally:
        if train_env is not None:
            train_env.close()
        if rclpy_inited and rclpy.ok():
            rclpy.shutdown()
