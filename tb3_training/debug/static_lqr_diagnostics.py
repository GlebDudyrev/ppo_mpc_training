from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
import torch
from stable_baselines3 import PPO

from tb3_training.registries.envs import get_env_spec


REF_NAMES = ["x_ref", "y_ref", "theta_ref", "v_ref", "omega_ref"]
ACT_NAMES = ["v", "omega"]


def _stats_dict(arr: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
    }


def _vector_stats(arr2d: np.ndarray, names: list[str]) -> dict[str, dict[str, Any]]:
    arr2d = np.asarray(arr2d, dtype=np.float64)
    return {name: _stats_dict(arr2d[:, i]) for i, name in enumerate(names)}


def _fraction(mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=np.float64)
    return float(mask.mean()) if mask.size else 0.0


@torch.no_grad()
def _forward_policy_outputs(model: PPO, obs_batch: np.ndarray) -> dict[str, np.ndarray]:
    policy = model.policy
    obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=model.device)

    features = policy.extract_features(obs_tensor)
    latent_pi = policy.mlp_extractor.forward_actor(features)

    action_net = policy.action_net
    if not hasattr(action_net, "reference_head") or not hasattr(action_net, "scale_reference"):
        raise RuntimeError("Loaded policy action_net does not expose reference_head/scale_reference. Is this a static_lqr model?")

    raw_reference = action_net.reference_head(latent_pi)
    scaled_reference = action_net.scale_reference(raw_reference)
    mean_action = action_net(latent_pi)

    return {
        "latent_pi": latent_pi.detach().cpu().numpy(),
        "raw_reference": raw_reference.detach().cpu().numpy(),
        "scaled_reference": scaled_reference.detach().cpu().numpy(),
        "mean_action": mean_action.detach().cpu().numpy(),
    }


def _collect_observations(env_name: str, episodes: int, max_steps: int, seed: int, use_policy_rollout: bool, model: PPO | None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    env = get_env_spec(env_name).make(mode="eval")
    obs_records: list[np.ndarray] = []
    ep_summaries: list[dict[str, Any]] = []

    try:
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            ep_return = 0.0
            steps = 0
            terminated = False
            truncated = False
            obs_records.append(np.asarray(obs, dtype=np.float32))

            while not (terminated or truncated) and steps < max_steps:
                if use_policy_rollout:
                    if model is None:
                        raise RuntimeError("use_policy_rollout=True requires a loaded model")
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)
                obs_records.append(np.asarray(obs, dtype=np.float32))
                ep_return += float(reward)
                steps += 1

            ep_summaries.append(
                {
                    "episode": ep,
                    "steps": steps,
                    "return": ep_return,
                    "success": bool(info.get("success", False)),
                    "collision": bool(info.get("collision", False)),
                    "timeout": bool(info.get("timeout", False)),
                    "distance_to_goal": float(info.get("distance_to_goal", 0.0)),
                    "scene_name": info.get("scene_name"),
                    "world_name": info.get("world_name"),
                }
            )
    finally:
        env.close()

    return np.asarray(obs_records, dtype=np.float32), ep_summaries


def run_diagnostics(
    model_path: str,
    env_name: str,
    episodes: int,
    max_steps: int,
    seed: int,
    use_policy_rollout: bool,
    output_json: str | None,
) -> dict[str, Any]:
    model = PPO.load(model_path)

    obs_batch, episode_summaries = _collect_observations(
        env_name=env_name,
        episodes=episodes,
        max_steps=max_steps,
        seed=seed,
        use_policy_rollout=use_policy_rollout,
        model=model,
    )

    outputs = _forward_policy_outputs(model, obs_batch)
    raw_ref = outputs["raw_reference"]
    scaled_ref = outputs["scaled_reference"]
    mean_action = outputs["mean_action"]

    v = mean_action[:, 0]
    omega = mean_action[:, 1]

    report: dict[str, Any] = {
        "model_path": model_path,
        "env_name": env_name,
        "episodes": episodes,
        "max_steps": max_steps,
        "seed": seed,
        "observation_count": int(obs_batch.shape[0]),
        "rollout_mode": "policy" if use_policy_rollout else "random",
        "episode_summaries": episode_summaries,
        "latent_pi_stats": {
            "dim": int(outputs["latent_pi"].shape[1]),
            "overall_mean": float(outputs["latent_pi"].mean()),
            "overall_std": float(outputs["latent_pi"].std()),
            "overall_min": float(outputs["latent_pi"].min()),
            "overall_max": float(outputs["latent_pi"].max()),
        },
        "raw_reference_stats": _vector_stats(raw_ref, REF_NAMES),
        "scaled_reference_stats": _vector_stats(scaled_ref, REF_NAMES),
        "mean_action_stats": _vector_stats(mean_action, ACT_NAMES),
        "diagnostics": {
            "omega_positive_fraction": _fraction(omega > 0.0),
            "omega_negative_fraction": _fraction(omega < 0.0),
            "omega_near_zero_fraction": _fraction(np.abs(omega) < 0.05),
            "v_positive_fraction": _fraction(v > 0.0),
            "v_negative_fraction": _fraction(v < 0.0),
            "v_zeroish_fraction": _fraction(np.abs(v) < 0.02),
            "v_saturation_fraction": _fraction(np.abs(v) >= 0.215),
            "omega_saturation_fraction": _fraction(np.abs(omega) >= 2.75),
            "turn_right_bias_score": float((omega > 0.05).mean() - (omega < -0.05).mean()),
        },
        "samples": {
            "raw_reference_first10": raw_ref[:10].tolist(),
            "scaled_reference_first10": scaled_ref[:10].tolist(),
            "mean_action_first10": mean_action[:10].tolist(),
        },
    }

    if output_json is not None:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose static_lqr reference/action bias")
    parser.add_argument("--model-path", required=True, help="Path to saved PPO model")
    parser.add_argument("--env", default="simple_env", help="Environment name from registry")
    parser.add_argument("--episodes", type=int, default=4, help="Number of eval episodes to collect observations from")
    parser.add_argument("--max-steps", type=int, default=128, help="Max steps per episode for observation collection")
    parser.add_argument("--seed", type=int, default=123, help="Base seed")
    parser.add_argument(
        "--use-policy-rollout",
        action="store_true",
        help="Collect observations by rolling out the loaded policy instead of random actions",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to save diagnostics JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    rclpy.init()
    try:
        report = run_diagnostics(
            model_path=args.model_path,
            env_name=args.env,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            use_policy_rollout=args.use_policy_rollout,
            output_json=args.output_json,
        )
    finally:
        if rclpy.ok():
            rclpy.shutdown()

    print(json.dumps(report["diagnostics"], indent=2))
    print("scaled_reference_stats:")
    print(json.dumps(report["scaled_reference_stats"], indent=2))
    print("mean_action_stats:")
    print(json.dumps(report["mean_action_stats"], indent=2))
    if args.output_json:
        print(f"Saved diagnostics to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
