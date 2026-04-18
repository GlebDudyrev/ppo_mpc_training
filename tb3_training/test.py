import torch
import rclpy
from stable_baselines3 import PPO
from tb3_training.registries.envs import get_env_spec

MODEL_PATH = "experiments/static_lqr/simple_env/train/version_0/final_model/model"

rclpy.init()

env = get_env_spec("simple_env")
env = env.make(mode="eval")
model = PPO.load(MODEL_PATH, env=env)

obs_list = []
obs, info = env.reset(seed=123)
obs_list.append(obs)

for _ in range(7):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    obs_list.append(obs)
    if terminated or truncated:
        obs, info = env.reset()
        obs_list.append(obs)

obs_batch = torch.as_tensor(obs_list, dtype=torch.float32, device=model.device)

policy = model.policy
policy.train()
policy.zero_grad(set_to_none=True)

dist = policy.get_distribution(obs_batch)

target_actions = torch.zeros((obs_batch.shape[0], 2), dtype=torch.float32, device=model.device)

loss = -dist.log_prob(target_actions).mean()
loss.backward()

print("synthetic loss:", float(loss.detach().cpu()))

for name, param in policy.named_parameters():
    if (
        "action_net" in name
        or "reference_head" in name
        or "q_raw" in name
        or "r_raw" in name
        or "qf_raw" in name
    ):
        grad = param.grad
        if grad is None:
            print(name, "grad=None")
        else:
            print(name, "grad_norm=", float(grad.norm().detach().cpu()))

env.close()
rclpy.shutdown()
