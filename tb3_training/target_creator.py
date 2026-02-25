#!/usr/bin/env python3
import random
import os
from copy import copy
from time import sleep
from functools import partial
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, SetEntityState
from std_msgs.msg import Bool


class TargetCreatorNode(Node):

    def __init__(self):
        super().__init__('target_creator')

        self.target_pose_ = Pose()
        self.target_exist_ = False

        self.subscription_create_ = self.create_subscription(
            Bool, '/target_creator', self.target_creator_callback, 10
        )
        self.subscription_update_ = self.create_subscription(
            Bool, '/target_updater', self.target_update_callback, 10
        )
        self.publisher_ = self.create_publisher(
            Pose, '/target_pose', 10
        )
        self.target_pose_timer_ = self.create_timer(
            0.2, self.target_pose_timer_callback
        )

        self.get_logger().info('TargetCreatorNode has been started')

    def target_pose_timer_callback(self):
        self.publisher_.publish(self.target_pose_)

    def target_update_callback(self, msg: Bool):
        if msg.data:
            if self.target_exist_:
                self.update_target()
            else:
                self.create_target()
        else:
            self.get_logger().warn('To update target need data == true')

    def update_target(self):
        client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
    
        while not client.wait_for_service(1):
            self.get_logger().warn('Waiting for service: /gazebo/set_entity_state')

        self.target_pose_.position.x = random.uniform(-2, 2)
        self.target_pose_.position.y = random.uniform(-2, 2)
        self.target_pose_.position.z = 0.5

        request = SetEntityState.Request()
        request.state.name = 'target'
        request.state.pose.position.x = self.target_pose_.position.x
        request.state.pose.position.y = self.target_pose_.position.y
        request.state.pose.position.z = self.target_pose_.position.z

        future = client.call_async(request)
        future.add_done_callback(partial(self.callback_update_target))

    def callback_update_target(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('Target update successfully')
            else:
                self.get_logger().warn('Target issue opdate')
        except Exception as e:
            self.get_logger().error(f'Error update target: {e}')

    def target_creator_callback(self, msg: Bool):
        if msg.data and not self.target_exist_:
            self.create_target()
        elif msg.data and self.target_exist_:
            self.get_logger().info('The goal already exists')
        elif msg.data and self.target_exist_:
            self.remove_target()
        else:
            self.get_logger().info('It is impossible to delete the model, it does not exist')

    def create_target(self):
        client = self.create_client(SpawnEntity, '/spawn_entity')

        while not client.wait_for_service(1):
            self.get_logger().warn('Waiting for service: /spawn_entity')

        self.target_pose_.position.x = random.uniform(-2, 2)
        self.target_pose_.position.y = random.uniform(-2, 2)
        self.target_pose_.position.z = 0.5

        request = SpawnEntity.Request()
        request.name = 'target'
        request.xml = self.get_target_model()
        request.initial_pose.position.x = self.target_pose_.position.x
        request.initial_pose.position.y = self.target_pose_.position.y
        request.initial_pose.position.z = self.target_pose_.position.z

        future = client.call_async(request)
        future.add_done_callback(partial(self.callback_create_target))

    def callback_create_target(self, future):
        try:
            response = future.result()
            if response.success:
                self.target_exist_ = True
                self.get_logger().info('Target created successfully')
            else:
                self.target_exist_ = True
                self.get_logger().warn('Target issue created')
        except Exception as e:
            self.get_logger().error(f'Error create target: {e}')

    def remove_target(self):
        client = self.create_client(DeleteEntity, '/delete_entity')

        while not client.wait_for_service(1):
            self.get_logger().warn('Waiting for service: /delete_entity')

        request = DeleteEntity.Request()
        request.name = 'target'

        future = client.call_async(request)
        future.add_done_callback(partial(self.callback_remove_target))

    def callback_remove_target(self, future):
        try:
            response = future.result()
            if response.success:
                self.target_exist_ = False
                self.get_logger().info('Target removed successfully')
            else:
                self.target_exist_ = False
                self.get_logger().warn('Target issue removed')
        except Exception as e:
            self.get_logger().error(f'Error remove target: {e}')

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


def main(args=None):
    rclpy.init(args=args)

    node = TargetCreatorNode()
    rclpy.spin(node)

    rclpy.shutdown()
