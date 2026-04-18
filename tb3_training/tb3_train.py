#!/usr/bin/env python3
from __future__ import annotations

import argparse

from tb3_training.registries import list_envs, list_models
from tb3_training.runners.train_runner import TrainRunConfig, run_train


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Universal TurtleBot3 training runner")
    parser.add_argument("--model", choices=list_models(), default="pure_rl")
    parser.add_argument("--env", choices=list_envs(), default="simple_env")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--experiments-root", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--validate-gradients", type=bool, default=False)

    return parser


def main(args: list[str] | None = None) -> int:
    parsed = build_arg_parser().parse_args(args=args)
    config = TrainRunConfig(
        model_name=parsed.model,
        env_name=parsed.env,
        total_timesteps=parsed.total_timesteps,
        learning_rate=parsed.learning_rate,
        n_steps=parsed.n_steps,
        batch_size=parsed.batch_size,
        n_epochs=parsed.n_epochs,
        gamma=parsed.gamma,
        seed=parsed.seed,
        checkpoint_freq=parsed.checkpoint_freq,
        experiments_root=parsed.experiments_root,
        progress_bar=not parsed.no_progress_bar,
        device=parsed.device,
        validate_gradients=parsed.validate_gradients
    )
    return run_train(config)


if __name__ == "__main__":
    raise SystemExit(main())
