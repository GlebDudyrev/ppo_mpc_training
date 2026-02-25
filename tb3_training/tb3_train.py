#!/usr/bin/env Python3
import os
import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from .training_env import TurtleBot3Env

def main(args=None):
    rclpy.init(args=args)

    home_path = os.path.expanduser('~')
    base_dir = os.path.join(home_path, 'turtlebot3_rl_results')
    log_dir = os.path.join(base_dir, 'logs')
    save_dir = os.path.join(base_dir, 'models')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    env = TurtleBot3Env()

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log=log_dir
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_dir,
        name_prefix="ppo_turtlebot3_model"
    )

    print(f"Обучение началось! Логи пишутся в: {log_dir}")
    try:
        model.learn(
            total_timesteps=200000, 
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        model.save(os.path.join(save_dir, "final_model"))
        print("Обучение завершено успешно!")

    except KeyboardInterrupt:
        print("Обучение превано пользователем. Сохраняю текущие веса...")
        model.save(os.path.join(save_dir, "interrupted_model"))

    finally:
        env.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()