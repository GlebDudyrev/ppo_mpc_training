#!/usr/bin/env Python3
import rclpy
from training_env import TurtleBot3Env


def main():
    rclpy.init()
    env = TurtleBot3Env()
    
    for episode in range(3):
        obs, info = env.reset()
        done = False
        step_count = 0
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Episode: {episode}, Step: {step_count}, Reward: {reward:.2f}")
            
            done = terminated or truncated
            step_count += 1
            
    env.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
