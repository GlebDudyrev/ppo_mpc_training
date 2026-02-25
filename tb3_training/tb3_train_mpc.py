#!/usr/bin/env Python3
import os
import rclpy
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from .training_env import TurtleBot3Env
from .ppo_mpc_policy import MPCActorCriticPolicy 


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

    policy_kwargs = dict(
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=dict(eps=1e-5),
        net_arch=dict(pi=[], vf=[128, 64]),
    )

    model = PPO(
        policy=MPCActorCriticPolicy,
        env=env,
        ent_coef=0.2,
        learning_rate=2e-3,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs, # Передаем настройки
        device="cuda" if torch.cuda.is_available() else "cpu" # Рекомендуется для MPC
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
