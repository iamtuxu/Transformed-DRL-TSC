import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from networks import PolicyNetwork, ValueNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, clip_epsilon, gae_lambda, batch_size, update_epochs, network_file=''):
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(device)
        self.value_net = ValueNetwork(state_dim).to(device)
        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.update_epochs = update_epochs

        if network_file:
            self.policy_net.load_state_dict(torch.load(network_file, map_location=device))

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = self.policy_net(state)
            dist = Categorical(probs)
            action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, trajectories):
        states, actions, log_probs, rewards, next_states, dones = zip(*trajectories)
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions).to(device)
        log_probs = torch.tensor(log_probs).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        # 计算值函数
        values = self.value_net(states).squeeze().detach()  # 分离计算图
        next_values = self.value_net(torch.tensor(next_states, dtype=torch.float32).to(device)).squeeze().detach()
        advantages = self.compute_advantages(rewards, values, next_values, dones)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns = advantages + values  # 确保 values 已分离计算图

        for _ in range(self.update_epochs):
            indices = np.random.permutation(len(states))
            for i in range(0, len(states), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs = log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Policy loss
                new_probs = self.policy_net(batch_states)
                dist = Categorical(new_probs)
                new_log_probs = dist.log_prob(batch_actions)
                ratios = torch.exp(new_log_probs - batch_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(self.value_net(batch_states).squeeze(), batch_returns)

                # Update
                self.optimizer.zero_grad()
                (policy_loss + value_loss).backward()  # 不需要 retain_graph=True
                self.optimizer.step()