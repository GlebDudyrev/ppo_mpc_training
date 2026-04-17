#!/usr/bin/env python3
from __future__ import annotations

import argparse

from tb3_training.registries import list_envs, list_models
from tb3_training.runners.eval_runner import EvalRunConfig, run_eval


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Universal TurtleBot3 evaluation runner")
    parser.add_argument("--model", choices=list_models(), default="pure_rl")
    parser.add_argument("--env", choices=list_envs(), default="simple_env")
    parser.add_argument("--train-version", type=int, default=None, help="train version to evaluate; latest if omitted")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument("--experiments-root", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def main(args: list[str] | None = None) -> int:
    parsed = build_arg_parser().parse_args(args=args)
    config = EvalRunConfig(
        model_name=parsed.model,
        env_name=parsed.env,
        train_version=parsed.train_version,
        episodes=parsed.episodes,
        seed=parsed.seed,
        deterministic=parsed.deterministic,
        experiments_root=parsed.experiments_root,
        output_json=parsed.output_json,
        device=parsed.device,
    )
    return run_eval(config)


if __name__ == "__main__":
    raise SystemExit(main())
