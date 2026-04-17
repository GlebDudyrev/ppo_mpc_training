#!/usr/bin/env python3

"""Environment validation and smoke-test utility for TurtleBot3 RL training."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import rclpy
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from tb3_training.registries import get_env_spec, list_envs


def run_smoke_test(env, *, episodes: int, max_steps: int) -> None:
    print("[check_env] starting smoke test")
    episode_returns: list[float] = []

    for episode_idx in range(episodes):
        obs, info = env.reset()
        assert env.observation_space.contains(obs), "reset() observation outside observation_space"
        terminated = False
        truncated = False
        steps = 0
        ep_return = 0.0
        step_info = info

        while not (terminated or truncated) and steps < max_steps:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            assert env.observation_space.contains(next_obs), "step() observation outside observation_space"
            if not np.isfinite(reward):
                raise ValueError("Encountered non-finite reward during smoke test")
            steps += 1
            ep_return += float(reward)
            obs = next_obs

        episode_returns.append(ep_return)
        print(
            f"[check_env] episode={episode_idx + 1}/{episodes} "
            f"steps={steps} return={ep_return:.3f} "
            f"distance_to_goal={float(step_info.get('distance_to_goal', np.nan)):.3f} "
            f"min_lidar_distance={float(step_info.get('min_lidar_distance', np.nan)):.3f} "
            f"collision={bool(step_info.get('collision', False))} "
            f"success={bool(step_info.get('success', False))}"
        )

    if episode_returns:
        print(
            "[check_env] smoke test complete: "
            f"mean_return={float(np.mean(episode_returns)):.3f}, "
            f"std_return={float(np.std(episode_returns)):.3f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a registered TurtleBot3 Gymnasium environment")
    parser.add_argument("--env", choices=list_envs(), default="simple_env", help="registered environment name")
    parser.add_argument("--mode", choices=["train", "eval"], default="train", help="initial environment mode")
    parser.add_argument("--episodes", type=int, default=2, help="number of smoke-test episodes")
    parser.add_argument("--max-steps", type=int, default=64, help="maximum random steps per smoke-test episode")
    parser.add_argument(
        "--skip-sb3-check",
        action="store_true",
        help="skip formal Stable-Baselines3 env checker and only run smoke test",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    env = None
    rclpy_inited = False
    try:
        if not rclpy.ok():
            rclpy.init(args=None)
            rclpy_inited = True

        env_spec = get_env_spec(args.env)
        print(f"[check_env] env={env_spec.name} world={env_spec.world_name} mode={args.mode}")
        env = env_spec.make(mode=args.mode)

        if not args.skip_sb3_check:
            print("[check_env] running Stable-Baselines3 env checker")
            sb3_check_env(env, warn=True, skip_render_check=True)
            print("[check_env] Stable-Baselines3 env checker passed")

        run_smoke_test(env, episodes=args.episodes, max_steps=args.max_steps)
        print("[check_env] validation finished successfully")
        return 0
    except KeyboardInterrupt:
        print("[check_env] interrupted by user")
        return 130
    except Exception as exc:
        print(f"[check_env] validation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if env is not None:
            env.close()
        if rclpy_inited and rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
