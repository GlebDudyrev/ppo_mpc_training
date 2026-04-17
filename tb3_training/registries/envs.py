from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional

from tb3_training.training_env import TurtleBot3Env


EvalCase = dict[str, float]
Bounds = tuple[float, float]


@dataclass(frozen=True)
class EnvSpec:
    """Named environment configuration used by universal train/eval runners.

    The Gazebo world is expected to be launched externally.  The spec records
    which world should be running and provides spawn/evaluation rules for the
    single TurtleBot3Env class.
    """

    name: str
    description: str
    world_name: str
    factory: Callable[..., TurtleBot3Env] = TurtleBot3Env
    train_kwargs: dict = field(default_factory=dict)
    eval_kwargs: dict = field(default_factory=dict)
    eval_episodes: list[EvalCase] = field(default_factory=list)

    def make(self, mode: str, **overrides) -> TurtleBot3Env:
        if mode not in {"train", "eval"}:
            raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")

        kwargs = dict(self.train_kwargs if mode == "train" else self.eval_kwargs)
        kwargs.update(overrides)
        kwargs.setdefault("scene_name", self.name)
        kwargs.setdefault("world_name", self.world_name)
        if mode == "eval":
            kwargs.setdefault("eval_episodes", self.eval_episodes)
        kwargs["mode"] = mode
        return self.factory(**kwargs)

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "world_name": self.world_name,
            "train_kwargs": self.train_kwargs,
            "eval_kwargs": self.eval_kwargs,
            "eval_episodes": self.eval_episodes,
        }


_ENV_REGISTRY: Dict[str, EnvSpec] = {}


def register_env(spec: EnvSpec) -> None:
    if spec.name in _ENV_REGISTRY:
        raise KeyError(f"Environment already registered: {spec.name}")
    _ENV_REGISTRY[spec.name] = spec


def get_env_spec(name: str) -> EnvSpec:
    try:
        return _ENV_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_ENV_REGISTRY)) or "<empty>"
        raise KeyError(f"Unknown environment '{name}'. Available: {available}") from exc


def list_envs() -> list[str]:
    return sorted(_ENV_REGISTRY)


def _case(robot_x: float, robot_y: float, target_x: float, target_y: float, robot_yaw: float = 0.0) -> EvalCase:
    return {
        "robot_x": float(robot_x),
        "robot_y": float(robot_y),
        "robot_yaw": float(robot_yaw),
        "target_x": float(target_x),
        "target_y": float(target_y),
    }


# Stage 1 is an empty world.  These cases cover forward, backward, lateral and
# diagonal goal configurations with different initial headings.
SIMPLE_EVAL_CASES: list[EvalCase] = [
    _case(-1.5, 0.0, 1.5, 0.0, 0.0),
    _case(1.5, 0.0, -1.5, 0.0, 3.1416),
    _case(0.0, -1.5, 0.0, 1.5, 1.5708),
    _case(0.0, 1.5, 0.0, -1.5, -1.5708),
    _case(-1.3, -1.3, 1.3, 1.3, 0.7854),
    _case(1.3, -1.3, -1.3, 1.3, 2.3562),
    _case(-1.3, 1.3, 1.3, -1.3, -0.7854),
    _case(1.3, 1.3, -1.3, -1.3, -2.3562),
]


# Stage 2 has four static columns in the common TurtleBot3 DQN setup.  The exact
# obstacle coordinates can vary by TurtleBot3 version, so the cases are kept in
# broad free regions and should be visually checked once in Gazebo.
STATIC_OBSTACLE_EVAL_CASES: list[EvalCase] = [
    _case(-1.8, 0.0, 1.8, 0.0, 0.0),
    _case(1.8, 0.0, -1.8, 0.0, 3.1416),
    _case(0.0, -1.8, 0.0, 1.8, 1.5708),
    _case(0.0, 1.8, 0.0, -1.8, -1.5708),
    _case(-1.8, -1.2, 1.8, 1.2, 0.5),
    _case(1.8, -1.2, -1.8, 1.2, 2.6),
    _case(-1.8, 1.2, 1.8, -1.2, -0.5),
    _case(1.8, 1.2, -1.8, -1.2, -2.6),
    _case(-1.2, -1.8, 1.2, 1.8, 0.9),
    _case(1.2, -1.8, -1.2, 1.8, 2.2),
]


# Stage 4 is intentionally a harder held-out evaluation environment.  Keep it out
# of the first training comparison unless all methods already work on stage 1/2.
HARD_EVAL_CASES: list[EvalCase] = [
    _case(-1.8, 0.0, 1.8, 0.0, 0.0),
    _case(1.8, 0.0, -1.8, 0.0, 3.1416),
    _case(-1.8, -1.4, 1.8, 1.4, 0.6),
    _case(1.8, -1.4, -1.8, 1.4, 2.5),
    _case(-1.6, 1.6, 1.6, -1.6, -0.7),
    _case(1.6, 1.6, -1.6, -1.6, -2.4),
]


COMMON_TASK_KWARGS = {
    "max_steps": 512,
    "reset_min_lidar_distance": 0.22,
    "min_start_goal_dist": 1.0,
    "max_start_goal_dist": 3.8,
}


register_env(
    EnvSpec(
        name="simple_env",
        description="TurtleBot3 DQN stage1: empty world, go-to-goal sanity benchmark.",
        world_name="turtlebot3_dqn_stage1",
        train_kwargs={
            **COMMON_TASK_KWARGS,
            "train_robot_bounds": ((-1.8, 1.8), (-1.8, 1.8)),
            "train_target_bounds": ((-1.8, 1.8), (-1.8, 1.8)),
        },
        eval_kwargs={**COMMON_TASK_KWARGS},
        eval_episodes=SIMPLE_EVAL_CASES,
    )
)

register_env(
    EnvSpec(
        name="static_obstacle_env",
        description="TurtleBot3 DQN stage2: static columns, main obstacle-avoidance benchmark.",
        world_name="turtlebot3_dqn_stage2",
        train_kwargs={
            **COMMON_TASK_KWARGS,
            "train_robot_bounds": ((-1.9, 1.9), (-1.9, 1.9)),
            "train_target_bounds": ((-1.9, 1.9), (-1.9, 1.9)),
        },
        eval_kwargs={**COMMON_TASK_KWARGS},
        eval_episodes=STATIC_OBSTACLE_EVAL_CASES,
    )
)

register_env(
    EnvSpec(
        name="hard_env",
        description="TurtleBot3 DQN stage4: hard held-out evaluation world.",
        world_name="turtlebot3_dqn_stage4",
        train_kwargs={
            **COMMON_TASK_KWARGS,
            "train_robot_bounds": ((-1.8, 1.8), (-1.8, 1.8)),
            "train_target_bounds": ((-1.8, 1.8), (-1.8, 1.8)),
        },
        eval_kwargs={**COMMON_TASK_KWARGS},
        eval_episodes=HARD_EVAL_CASES,
    )
)
