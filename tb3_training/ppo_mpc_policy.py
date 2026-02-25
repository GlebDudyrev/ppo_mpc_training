import torch
import torch.nn as nn
import numpy as np
import math
from stable_baselines3.common.policies import ActorCriticPolicy

class DifferentiableMPCLayer(nn.Module):
    def __init__(self, horizon=10, dt=0.1):
        super().__init__()
        self.H = horizon
        self.dt = dt
        self.max_v = 0.22
        self.max_w = 2.8

    def compute_cost(self, state, action_seq, obstacles, target_pos, weights):
        batch_size = state.shape[0]

        total_cost = (weights['dist'] + weights['obs'] + weights['angle'] + weights['rv'] + weights['rw']).sum() * 0.0
        
        total_cost = total_cost.repeat(batch_size)
        curr_state = state.clone()

        for t in range(self.H):
            v_raw = torch.tanh(action_seq[:, t, 0])
            w_raw = torch.tanh(action_seq[:, t, 1])
            
            v = (v_raw + 1.0) / 2.0 * self.max_v
            w = w_raw * self.max_w
            
            new_x = curr_state[:, 0] + v * torch.cos(curr_state[:, 2]) * self.dt
            new_y = curr_state[:, 1] + v * torch.sin(curr_state[:, 2]) * self.dt
            new_yaw = curr_state[:, 2] + w * self.dt
            curr_state = torch.stack([new_x, new_y, new_yaw], dim=1)

            dist_to_goal = torch.norm(curr_state[:, :2] - target_pos, dim=1)
            
            target_dir = target_pos - curr_state[:, :2]
            target_angle = torch.atan2(target_dir[:, 1], target_dir[:, 0])
            angle_err = (torch.atan2(torch.sin(target_angle - curr_state[:, 2]), 
                                   torch.cos(target_angle - curr_state[:, 2])))

            dist_to_obs = torch.norm(curr_state[:, :2].unsqueeze(1) - obstacles, dim=2)
            min_dist = torch.min(dist_to_obs, dim=1)[0]
            collision_cost = torch.clamp(0.4 - min_dist, min=0)**2

            total_cost += dist_to_goal ** 2 * weights['dist'].flatten() * 10.0
            total_cost += angle_err ** 2 * weights['angle'].flatten() * 5.0
            total_cost += collision_cost ** 2 * weights['obs'].flatten() * 100.0
            total_cost += (v**2 * weights['rv'].flatten() * 1.0)
            total_cost += (w**2 * weights['rw'].flatten() * 0.1)

        return total_cost.mean()

    @torch.enable_grad()
    def forward(self, initial_state, scan_data, target_pos, weights):
        device = initial_state.device
        batch_size = initial_state.shape[0]
        
        action_seq = torch.zeros(batch_size, self.H, 2, device=device).requires_grad_(True)
        
        angles = torch.linspace(0, 2 * math.pi, 36, device=device)
        obs_x = scan_data * torch.cos(angles)
        obs_y = scan_data * torch.sin(angles)
        obstacles = torch.stack([obs_x, obs_y], dim=2)

        lr_inner = 0.2
        for _ in range(15):
            cost = self.compute_cost(initial_state, action_seq, obstacles, target_pos, weights)
            
            grads = torch.autograd.grad(
                cost, action_seq, 
                create_graph=True, 
                retain_graph=True,
                allow_unused=True
            )[0]
            
            if grads is not None:
                action_seq = action_seq - lr_inner * grads

        best_action_raw = action_seq[:, 0, :]
        v_final = (torch.tanh(best_action_raw[:, 0]) + 1.0) / 2.0 * self.max_v
        w_final = torch.tanh(best_action_raw[:, 1]) * self.max_w
        
        return torch.stack([v_final, w_final], dim=1)

class WeightNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
            nn.Softplus()
        )

    def forward(self, obs):
        return self.net(obs) + 0.05

class MPCActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_net = WeightNet(self.observation_space.shape[0]).to(self.device)
        self.mpc_layer = DifferentiableMPCLayer(horizon=10, dt=0.1).to(self.device)
        self.log_std = nn.Parameter(torch.ones(self.action_space.shape[0]) * -0.5)

    def forward(self, obs, deterministic=False):
        with torch.enable_grad():
            w_raw = self.weight_net(obs)
            weights = {
                'dist': w_raw[:, 0:1], 'obs': w_raw[:, 1:2],
                'rv': w_raw[:, 2:3], 'angle': w_raw[:, 3:4], 'rw': w_raw[:, 4:5]
            }
            
            scan_data = obs[:, :36]
            dist = obs[:, 36]
            angle = obs[:, 37]
            target_pos = torch.stack([dist * torch.cos(angle), dist * torch.sin(angle)], dim=1)
            initial_state = torch.zeros(obs.shape[0], 3, device=obs.device)
            
            mean_actions = self.mpc_layer(initial_state, scan_data, target_pos, weights)

        weigths_info = '\n'.join([f'{key.upper()}: {value}' for key, value in weights.items()])
        print(f" >>> MPC WEIGHTS: \n{weigths_info}")

        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        actions = distribution.mode() if deterministic else distribution.sample()
        
        return actions, self.predict_values(obs), distribution.log_prob(actions)

    def _predict(self, observation, deterministic=False):
        return self.forward(observation, deterministic)[0]

    def evaluate_actions(self, obs, actions):
        w_raw = self.weight_net(obs)
        weights = {
            'dist': w_raw[:, 0:1], 'obs': w_raw[:, 1:2],
            'rv': w_raw[:, 2:3], 'angle': w_raw[:, 3:4], 'rw': w_raw[:, 4:5]
        }
        target_pos = torch.stack([obs[:, 36] * torch.cos(obs[:, 37]), obs[:, 36] * torch.sin(obs[:, 37])], dim=1)
        mean_actions = self.mpc_layer(torch.zeros(obs.shape[0], 3, device=obs.device), obs[:, :36], target_pos, weights)

        grads = []
        for param in self.weight_net.parameters():
            if param.grad is not None:
                grads.append(param.grad.norm().item())
        
        if len(grads) > 0:
            avg_grad = sum(grads) / len(grads)
            print(f" >>> GRADIENT UPDATE | Avg Param Grad Norm: {avg_grad:.6f}")
        
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        return self.predict_values(obs), distribution.log_prob(actions), distribution.entropy()
