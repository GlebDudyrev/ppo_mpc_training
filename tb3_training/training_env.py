#!/usr/bin/env python3
import random
import math
from functools import partial
from time import sleep
from copy import copy
import os
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from gazebo_msgs.srv import SetEntityState, SpawnEntity
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TurtleBot3Env(gym.Env, Node):
    def __init__(self):
        super(TurtleBot3Env, self).__init__('training_env')

        self.action_space = spaces.Box(
            low=np.array([0.0, -2.8]), 
            high=np.array([0.22, 2.8]), 
            dtype=np.float32
        )

        low = np.array([0.0] * 36 + [0.0] + [-1.0], dtype=np.float32)
        high = np.array([1.0] * 36 + [1.0] + [1.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32
        )

        self.current_step = 0
        self.max_steps = 512

        self.target_x = 0
        self.target_y = 0
        self.robot_x = 0
        self.robot_y = 0
        self.robot_yaw = 0
        self.scan_data = np.ones(36, dtype=np.float32)
        self.prev_dist = 0

        self.target_spawned_ = False

        self.spawn_client = self.create_client(
            SpawnEntity, '/spawn_entity'
        )
        while not self.spawn_client.wait_for_service(1):
            self.get_logger().warn('Waiting for service: /spawn_entity')

        self.set_state_client_ = self.create_client(
            SetEntityState, '/gazebo/set_entity_state'
        )
        while not self.set_state_client_.wait_for_service(1):
            self.get_logger().warn('Waiting for service: /set_entity_state')

        self.cmd_vel_publisher_ = self.create_publisher(
            Twist, '/cmd_vel', 10
        )

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )
        self.scan_subscription_ = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile
        )

        self.odom_subscription_ = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

    def scan_callback(self, scan: LaserScan):
        scan = np.array(scan.ranges)
        scan = np.nan_to_num(scan, nan=3.5, posinf=3.5, neginf=3.5)
        scan = np.clip(scan, 0.01, 3.5)

        indices = np.linspace(0, len(scan) - 1, 36, dtype=int)
        reduced_scan = scan[indices]

        normalized_scan = reduced_scan / 3.5
        
        self.scan_data = normalized_scan.astype(np.float32)

    def odom_callback(self, odom: Odometry):
        self.robot_x = odom.pose.pose.position.x
        self.robot_y = odom.pose.pose.position.y

        q = odom.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

    def callback_reset_robot(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('Robot reset successfully')
            else:
                self.get_logger().warn('Robot issue reset')
        except Exception as e:
            self.get_logger().error(f'Error reset robot: {e}')

    def get_normalized_distance(self):        
        distance = self.get_distance()
        
        max_dist = 5.0
        normalized_distance = distance / max_dist
        
        return np.clip(normalized_distance, 0.0, 1.0)
    
    def get_distance(self):
        dx = self.target_x - self.robot_x
        dy = self.target_y - self.robot_y
        
        distance = math.hypot(dx, dy)
        
        return float(distance)
        
    def get_relative_angle(self):
        goal_angle = math.atan2(
            self.target_y - self.robot_y,
            self.target_x - self.robot_x
        )

        heading = goal_angle - self.robot_yaw

        if heading > math.pi:
            heading -= 2 * math.pi
        elif heading < -math.pi:
            heading += 2 * math.pi

        return heading / math.pi
    
    def get_observation(self):
        while rclpy.spin_once(self, timeout_sec=0):
            pass
        
        scan = self.scan_data
        dist = self.get_normalized_distance()
        angle = self.get_relative_angle()

        return np.array([*scan, dist, angle], dtype=np.float32)
    
    def teleport_entity(self, name, x, y, z=None, need_twisted=False):
        request = SetEntityState.Request()
        request.state.name = name
        request.state.pose.position.x = x
        request.state.pose.position.y = y

        if z is not None:
            request.state.pose.position.z = z

        if need_twisted:
            request.state.twist.linear.z = 0.001

        future = self.set_state_client_.call_async(request)
        future.add_done_callback(partial(self.callback_teleport_entity))

    def callback_teleport_entity(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('Entity teleport successfully')
            else:
                self.get_logger().warn('Entity issue teleport')
        except Exception as e:
            self.get_logger().error(f'Error teleport Entity: {e}')

    def spawn_target(self, x, y, name='target'):
        request = SpawnEntity.Request()
        request.name = name
        request.xml = self.get_target_model()
        request.initial_pose.position.x = x
        request.initial_pose.position.y = y
        request.initial_pose.position.z = 0.5

        future = self.spawn_client.call_async(request)
        future.add_done_callback(partial(self.callback_create_target))

    def callback_create_target(self, future):
        try:
            response = future.result()
            if response.success:
                self.target_spawned_ = True
                self.get_logger().info('Target created successfully')
            else:
                self.target_spawned_ = True
                self.get_logger().warn('Target issue created')
        except Exception as e:
            self.get_logger().error(f'Error create target: {e}')

    def get_target_model(self):
        self.model_path = os.path.join(
            get_package_share_directory('tb3_training'),
            'models',
            'target.sdf'
        )

        try:
            with open(self.model_path, 'r') as f:
                model = f.read()
                return model
        except FileNotFoundError:
            self.get_logger().error(f"Файл модели не найден по пути: {self.model_path}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        stop_msg = Twist()
        self.cmd_vel_publisher_.publish(stop_msg)

        self.target_x = random.uniform(1.1, 2.0) * random.choice([-1, 1])
        self.target_y = random.uniform(1.1, 2.0) * random.choice([-1, 1])
        self.robot_x = random.uniform(-0.5, 0.5)
        self.robot_y = random.uniform(-0.5, 0.5)

        self.teleport_entity('burger', self.robot_x, self.robot_y)

        if self.target_spawned_:
            self.teleport_entity('target', self.target_x, self.target_y, 0.5)
        else:
            self.spawn_target(self.target_x, self.target_y)

        self.scan_data = None
        start_wait = self.get_clock().now()
        while self.scan_data is None:
            rclpy.spin_once(self, timeout_sec=0.01)
            if (self.get_clock().now() - start_wait).nanoseconds > 1e9:
                self.get_logger().warn("Таймаут ожидания скана!")
                break

        observation = self.get_observation()
        self.prev_dist = self.get_distance()

        self.current_step = 0

        return observation, {}
    
    def apply_action(self, action):
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = float(action[0])
        cmd_vel_msg.angular.z = float(action[1])

        self.cmd_vel_publisher_.publish(cmd_vel_msg)

    def calculate_reward(self, current_dist, min_laser_dist, action):
        if min_laser_dist < 0.05:
            return -100.0, True

        if current_dist < 0.15:
            return 150.0, True

        reward = (self.prev_dist - current_dist) * 150

        if min_laser_dist < 0.3:
            reward -= (0.3 - min_laser_dist) * 5.0 

        if abs(action[0]) < 0.05:
            reward -= 0.1
        else:
            reward += 0.05
        
        return reward, False

    def step(self, action):
        self.current_step += 1

        self.apply_action(action)

        sleep(0.1) 

        self.apply_action((0, 0))

        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0)

        new_obs = self.get_observation()
        
        min_laser_dist = np.min(self.scan_data)
        current_dist = self.get_distance()
        reward, terminated = self.calculate_reward(current_dist, min_laser_dist, action)

        truncated = False
        if self.current_step > self.max_steps:
            truncated = True

        self.prev_dist = current_dist
        
        return new_obs, reward, terminated, truncated, {}

    def close(self):
        pass
