#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import time
from functools import partial
from typing import Optional

import gymnasium as gym
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.srv import SetEntityState, SpawnEntity
from geometry_msgs.msg import Twist
from gymnasium import spaces
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan


class TurtleBot3Env(gym.Env, Node):
    """ROS2/Gazebo + Gymnasium environment for TurtleBot3 goal reaching.

    Frozen task contract for pure RL, RL + static LQR, and RL + learned LQR:
      * action = commanded (v, w), applied with sample-and-hold semantics;
      * observation = compact lidar + goal geometry + current robot velocities;
      * reward = progress + success - collision - safety - smoothness - step;
      * scene geometry comes from an externally launched Gazebo world.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        control_dt: float = 0.10,
        lidar_num_beams: int = 36,
        lidar_max_range: float = 3.5,
        goal_norm_max_dist: float = 5.0,
        max_linear_velocity: float = 0.22,
        max_angular_velocity: float = 2.8,
        max_steps: int = 512,
        success_distance: float = 0.15,
        collision_distance: float = 0.15,
        safety_distance: float = 0.30,
        progress_reward_scale: float = 150.0,
        goal_reward: float = 150.0,
        collision_penalty: float = 100.0,
        safety_penalty_scale: float = 5.0,
        action_smoothness_scale: float = 0.10,
        step_penalty: float = 0.01,
        mode: str = "train",
        scene_name: str = "simple_env",
        world_name: str = "turtlebot3_dqn_stage1",
        train_robot_bounds: tuple[tuple[float, float], tuple[float, float]] = ((-0.5, 0.5), (-0.5, 0.5)),
        train_target_bounds: tuple[tuple[float, float], tuple[float, float]] = ((-2.0, 2.0), (-2.0, 2.0)),
        min_start_goal_dist: float = 1.0,
        max_start_goal_dist: Optional[float] = None,
        reset_min_lidar_distance: float = 0.20,
        reset_resample_attempts: int = 20,
        eval_episodes: Optional[list[dict]] = None,
    ):
        Node.__init__(self, "training_env")
        gym.Env.__init__(self)

        if mode not in {"train", "eval"}:
            raise ValueError(f"Unsupported mode: {mode}")

        self.control_dt = float(control_dt)
        self.lidar_num_beams = int(lidar_num_beams)
        self.lidar_max_range = float(lidar_max_range)
        self.goal_norm_max_dist = float(goal_norm_max_dist)
        self.max_linear_velocity = float(max_linear_velocity)
        self.max_angular_velocity = float(max_angular_velocity)
        self.max_steps = int(max_steps)
        self.success_distance = float(success_distance)
        self.collision_distance = float(collision_distance)
        self.safety_distance = float(safety_distance)

        self.progress_reward_scale = float(progress_reward_scale)
        self.goal_reward = float(goal_reward)
        self.collision_penalty = float(collision_penalty)
        self.safety_penalty_scale = float(safety_penalty_scale)
        self.action_smoothness_scale = float(action_smoothness_scale)
        self.step_penalty = float(step_penalty)

        self.mode = mode
        self.scene_name = scene_name
        self.world_name = world_name
        self.train_robot_bounds = train_robot_bounds
        self.train_target_bounds = train_target_bounds
        self.min_start_goal_dist = float(min_start_goal_dist)
        self.max_start_goal_dist = None if max_start_goal_dist is None else float(max_start_goal_dist)
        self.reset_min_lidar_distance = float(reset_min_lidar_distance)
        self.reset_resample_attempts = int(reset_resample_attempts)
        self.eval_episodes = list(eval_episodes or [])
        self.eval_episode_index = 0

        self.action_space = spaces.Box(
            low=np.array([-self.max_linear_velocity, -self.max_angular_velocity], dtype=np.float32),
            high=np.array([self.max_linear_velocity, self.max_angular_velocity], dtype=np.float32),
            dtype=np.float32,
        )

        low = np.array([0.0] * self.lidar_num_beams + [0.0, -1.0, -1.0, -1.0], dtype=np.float32)
        high = np.array([1.0] * self.lidar_num_beams + [1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.current_step = 0
        self.target_x = 0.0
        self.target_y = 0.0
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.robot_linear_velocity = 0.0
        self.robot_angular_velocity = 0.0
        self.scan_data = np.ones(self.lidar_num_beams, dtype=np.float32)
        self.prev_dist = 0.0
        self.initial_dist = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.target_spawned_ = False

        self.spawn_client = self.create_client(SpawnEntity, "/spawn_entity")
        while not self.spawn_client.wait_for_service(1.0):
            self.get_logger().warn("Waiting for service: /spawn_entity")

        self.set_state_client_ = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        while not self.set_state_client_.wait_for_service(1.0):
            self.get_logger().warn("Waiting for service: /gazebo/set_entity_state")

        self.cmd_vel_publisher_ = self.create_publisher(Twist, "/cmd_vel", 10)

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        self.scan_subscription_ = self.create_subscription(LaserScan, "/scan", self.scan_callback, qos_profile)
        self.odom_subscription_ = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)

    # --------------------------
    # ROS callbacks and helpers
    # --------------------------
    def spin_some(self, timeout_sec: float = 0.0, repeat: int = 1) -> None:
        for _ in range(repeat):
            rclpy.spin_once(self, timeout_sec=timeout_sec)

    def stop_robot(self) -> None:
        self.cmd_vel_publisher_.publish(Twist())

    def scan_callback(self, scan: LaserScan) -> None:
        scan_values = np.asarray(scan.ranges, dtype=np.float32)
        scan_values = np.nan_to_num(
            scan_values,
            nan=self.lidar_max_range,
            posinf=self.lidar_max_range,
            neginf=self.lidar_max_range,
        )
        scan_values = np.clip(scan_values, 0.01, self.lidar_max_range)
        indices = np.linspace(0, len(scan_values) - 1, self.lidar_num_beams, dtype=int)
        self.scan_data = (scan_values[indices] / self.lidar_max_range).astype(np.float32)

    def _await_new_scan_data(self, timeout_sec: float = 1.0) -> None:
        self.scan_data = None
        wait_start = time.time()
        while self.scan_data is None and time.time() - wait_start < timeout_sec:
            self.spin_some(timeout_sec=0.01, repeat=1)

    def odom_callback(self, odom: Odometry) -> None:
        self.robot_x = float(odom.pose.pose.position.x)
        self.robot_y = float(odom.pose.pose.position.y)

        q = odom.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

        self.robot_linear_velocity = float(odom.twist.twist.linear.x)
        self.robot_angular_velocity = float(odom.twist.twist.angular.z)

    def callback_teleport_entity(self, future) -> None:
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Entity teleported successfully")
            else:
                self.get_logger().warn("Entity teleport failed")
        except Exception as exc:  # pragma: no cover
            self.get_logger().error(f"Error teleporting entity: {exc}")

    def callback_create_target(self, future) -> None:
        try:
            response = future.result()
            self.target_spawned_ = True
            if response.success:
                self.get_logger().info("Target created successfully")
            else:
                self.get_logger().info("Target already exists or spawn failed; reset() will teleport it")
        except Exception as exc:  # pragma: no cover
            self.get_logger().error(f"Error creating target: {exc}")
            self.target_spawned_ = True

    def teleport_entity(
        self,
        name: str,
        x: float,
        y: float,
        z: Optional[float] = None,
        yaw: Optional[float] = None,
    ) -> None:
        request = SetEntityState.Request()
        request.state.name = name
        request.state.pose.position.x = float(x)
        request.state.pose.position.y = float(y)
        if z is not None:
            request.state.pose.position.z = float(z)
        if yaw is not None:
            half_yaw = 0.5 * float(yaw)
            request.state.pose.orientation.x = 0.0
            request.state.pose.orientation.y = 0.0
            request.state.pose.orientation.z = math.sin(half_yaw)
            request.state.pose.orientation.w = math.cos(half_yaw)
        future = self.set_state_client_.call_async(request)
        future.add_done_callback(partial(self.callback_teleport_entity))

    def get_target_model(self) -> str:
        model_path = os.path.join(get_package_share_directory("tb3_training"), "models", "target.sdf")
        with open(model_path, "r", encoding="utf-8") as file:
            return file.read()

    def spawn_target(self, x: float, y: float, name: str = "target") -> None:
        request = SpawnEntity.Request()
        request.name = name
        request.xml = self.get_target_model()
        request.initial_pose.position.x = float(x)
        request.initial_pose.position.y = float(y)
        request.initial_pose.position.z = 0.5
        future = self.spawn_client.call_async(request)
        future.add_done_callback(partial(self.callback_create_target))

    # --------------------------
    # Task geometry and obs
    # --------------------------
    def get_distance(self) -> float:
        dx = self.target_x - self.robot_x
        dy = self.target_y - self.robot_y
        return float(math.hypot(dx, dy))

    def get_normalized_distance(self) -> float:
        distance = self.get_distance()
        return float(np.clip(distance / self.goal_norm_max_dist, 0.0, 1.0))

    def get_relative_angle(self) -> float:
        goal_angle = math.atan2(self.target_y - self.robot_y, self.target_x - self.robot_x)
        heading = goal_angle - self.robot_yaw
        if heading > math.pi:
            heading -= 2.0 * math.pi
        elif heading < -math.pi:
            heading += 2.0 * math.pi
        return float(heading / math.pi)

    def get_normalized_velocities(self) -> tuple[float, float]:
        norm_v = np.clip(self.robot_linear_velocity / max(self.max_linear_velocity, 1e-6), -1.0, 1.0)
        norm_w = np.clip(self.robot_angular_velocity / max(self.max_angular_velocity, 1e-6), -1.0, 1.0)
        return float(norm_v), float(norm_w)

    def get_observation(self) -> np.ndarray:
        self.spin_some(timeout_sec=0.0, repeat=3)
        dist = self.get_normalized_distance()
        angle = self.get_relative_angle()
        norm_v, norm_w = self.get_normalized_velocities()
        return np.array([*self.scan_data, dist, angle, norm_v, norm_w], dtype=np.float32)

    def current_min_lidar_distance(self) -> float:
        return float(np.min(self.scan_data) * self.lidar_max_range)

    # --------------------------
    # Episode sampling/reset
    # --------------------------
    def set_mode(self, mode: str) -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode

    def _sample_xy(self, bounds: tuple[tuple[float, float], tuple[float, float]]) -> tuple[float, float]:
        (x_min, x_max), (y_min, y_max) = bounds
        return float(self.np_random.uniform(x_min, x_max)), float(self.np_random.uniform(y_min, y_max))

    def _sample_train_episode(self) -> tuple[float, float, float, float, float]:
        for _ in range(100):
            robot_x, robot_y = self._sample_xy(self.train_robot_bounds)
            target_x, target_y = self._sample_xy(self.train_target_bounds)
            dist = math.hypot(target_x - robot_x, target_y - robot_y)
            if dist < self.min_start_goal_dist:
                continue
            if self.max_start_goal_dist is not None and dist > self.max_start_goal_dist:
                continue
            robot_yaw = float(self.np_random.uniform(-math.pi, math.pi))
            return robot_x, robot_y, robot_yaw, target_x, target_y

        # Safe fallback if bounds are too restrictive.
        robot_x, robot_y = self._sample_xy(self.train_robot_bounds)
        target_x, target_y = self._sample_xy(self.train_target_bounds)
        robot_yaw = float(self.np_random.uniform(-math.pi, math.pi))
        return robot_x, robot_y, robot_yaw, target_x, target_y

    def _sample_eval_episode(self) -> tuple[float, float, float, float, float]:
        if not self.eval_episodes:
            return self._sample_train_episode()
        scenario = self.eval_episodes[self.eval_episode_index % len(self.eval_episodes)]
        self.eval_episode_index += 1
        return (
            float(scenario["robot_x"]),
            float(scenario["robot_y"]),
            float(scenario.get("robot_yaw", 0.0)),
            float(scenario["target_x"]),
            float(scenario["target_y"]),
        )

    def _step_info(self, *, terminated: bool = False, truncated: bool = False, reward_info: Optional[dict] = None) -> dict:
        """Return instantaneous step-level facts for logging and benchmarking.

        Episode-level aggregates such as path length, average speed and action
        smoothness are intentionally computed outside the environment by the
        benchmark layer.
        """
        min_lidar = self.current_min_lidar_distance()
        info = {
            "mode": self.mode,
            "scene_name": self.scene_name,
            "world_name": self.world_name,
            "current_step": int(self.current_step),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "timeout": bool(truncated and not terminated),
            "distance_to_goal": float(self.get_distance()),
            "initial_distance_to_goal": float(self.initial_dist),
            "min_lidar_distance": float(min_lidar),
            "robot_x": float(self.robot_x),
            "robot_y": float(self.robot_y),
            "robot_yaw": float(self.robot_yaw),
            "target_x": float(self.target_x),
            "target_y": float(self.target_y),
            "linear_velocity": float(self.robot_linear_velocity),
            "angular_velocity": float(self.robot_angular_velocity),
        }
        if reward_info:
            info.update(reward_info)
        return info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        options = options or {}

        self.stop_robot()
        self.spin_some(timeout_sec=0.01, repeat=5)

        override_mode = options.get("mode")
        if override_mode is not None:
            self.set_mode(override_mode)

        if {"robot_x", "robot_y", "target_x", "target_y"}.issubset(options.keys()):
            robot_x = float(options["robot_x"])
            robot_y = float(options["robot_y"])
            robot_yaw = float(options.get("robot_yaw", 0.0))
            target_x = float(options["target_x"])
            target_y = float(options["target_y"])
        elif self.mode == "eval":
            robot_x, robot_y, robot_yaw, target_x, target_y = self._sample_eval_episode()
        else:
            robot_x, robot_y, robot_yaw, target_x, target_y = self._sample_train_episode()

        self.robot_x = robot_x
        self.robot_y = robot_y
        self.robot_yaw = robot_yaw
        self.target_x = target_x
        self.target_y = target_y

        self.teleport_entity("burger", self.robot_x, self.robot_y, yaw=self.robot_yaw)
        if self.target_spawned_:
            self.teleport_entity("target", self.target_x, self.target_y, 0.5)
        else:
            self.spawn_target(self.target_x, self.target_y)

        self._await_new_scan_data()

        # If a train reset puts the robot too close to an obstacle, resample a few times.
        if self.mode == "train":
            attempts = 0
            while self.current_min_lidar_distance() < self.reset_min_lidar_distance and attempts < self.reset_resample_attempts:
                robot_x, robot_y, robot_yaw, target_x, target_y = self._sample_train_episode()
                self.robot_x = robot_x
                self.robot_y = robot_y
                self.robot_yaw = robot_yaw
                self.target_x = target_x
                self.target_y = target_y
                self.teleport_entity("burger", self.robot_x, self.robot_y, yaw=self.robot_yaw)
                self.teleport_entity("target", self.target_x, self.target_y, 0.5)
                self._await_new_scan_data()
                attempts += 1

        self.prev_dist = self.get_distance()
        self.initial_dist = self.prev_dist
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.current_step = 0
        obs = self.get_observation()
        return obs, self._step_info()

    # --------------------------
    # Action, reward, step
    # --------------------------
    def apply_action(self, action: np.ndarray | tuple[float, float]) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(2)
        clipped = np.clip(action, self.action_space.low, self.action_space.high)
        cmd = Twist()
        cmd.linear.x = float(clipped[0])
        cmd.angular.z = float(clipped[1])
        self.cmd_vel_publisher_.publish(cmd)
        return clipped

    def calculate_reward(
        self,
        *,
        current_dist: float,
        min_laser_dist: float,
        action: np.ndarray,
    ) -> tuple[float, bool, dict, float]:
        success = current_dist < self.success_distance
        collision = min_laser_dist < self.collision_distance

        progress = (self.prev_dist - current_dist) * self.progress_reward_scale
        safety_penalty = 0.0
        if min_laser_dist < self.safety_distance:
            safety_penalty = (self.safety_distance - min_laser_dist) * self.safety_penalty_scale

        action_delta = np.asarray(action, dtype=np.float32) - self.prev_action
        action_delta_sq = float(np.dot(action_delta, action_delta))
        smoothness_penalty = self.action_smoothness_scale * action_delta_sq

        reward = progress - safety_penalty - smoothness_penalty - self.step_penalty
        if success:
            reward += self.goal_reward
        if collision:
            reward -= self.collision_penalty

        info = {
            "progress_reward": float(progress),
            "safety_penalty": float(safety_penalty),
            "smoothness_penalty": float(smoothness_penalty),
            "step_penalty": float(self.step_penalty),
            "success": bool(success),
            "collision": bool(collision),
            "action_delta_sq": float(action_delta_sq),
        }
        terminated = success or collision
        return float(reward), terminated, info, action_delta_sq

    def step(self, action):
        self.current_step += 1
        clipped_action = self.apply_action(action)

        end_time = time.time() + self.control_dt
        while time.time() < end_time:
            self.spin_some(timeout_sec=0.01, repeat=1)

        new_obs = self.get_observation()
        min_laser_dist = self.current_min_lidar_distance()
        current_dist = self.get_distance()

        reward, terminated, reward_info, action_delta_sq = self.calculate_reward(
            current_dist=current_dist,
            min_laser_dist=min_laser_dist,
            action=clipped_action,
        )

        truncated = self.current_step >= self.max_steps
        self.prev_dist = current_dist
        self.prev_action = clipped_action.copy()

        info = self._step_info(terminated=terminated, truncated=truncated, reward_info=reward_info)
        info.update(
            {
                "action": clipped_action.astype(np.float32),
                "commanded_linear_velocity": float(clipped_action[0]),
                "commanded_angular_velocity": float(clipped_action[1]),
                "action_delta_sq": float(action_delta_sq),
            }
        )
        if terminated or truncated:
            self.stop_robot()
        return new_obs, reward, terminated, truncated, info

    def close(self):
        self.stop_robot()
        self.spin_some(timeout_sec=0.01, repeat=3)
