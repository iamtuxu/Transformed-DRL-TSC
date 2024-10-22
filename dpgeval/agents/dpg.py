import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


# class PolicyNetwork(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(PolicyNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, output_dim)
#
#         # Initialization can be customized if needed, e.g., Xavier:
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.xavier_uniform_(self.fc2.weight)
#         nn.init.xavier_uniform_(self.fc3.weight)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return F.softmax(x, dim=-1)  # Softmax for action selection


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()

        # Define the two-layer network with 512 units each
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)

        # Initialize weights with uniform distribution based on the input dimensions
        c = np.sqrt(1 / input_dim)
        nn.init.uniform_(self.fc1.weight, -c, c)
        nn.init.uniform_(self.fc1.bias, -c, c)
        nn.init.uniform_(self.fc2.weight, -c, c)
        nn.init.uniform_(self.fc2.bias, -c, c)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)  # Softmax for action selection


class DPGAgent:
    def __init__(self, input_dim, output_dim, gamma=0.95, lr=0.001, network_file=None):
        self.policy_network = PolicyNetwork(input_dim, output_dim).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.gamma = gamma

        # Load network weights if the file is provided
        if network_file:
            self.load_weights(network_file)

    def load_weights(self, file_path):
        try:
            self.policy_network.load_state_dict(torch.load(file_path, map_location=device))
            print(f'Loaded weights from {file_path}')
        except Exception as e:
            print(f'Error loading weights from {file_path}: {e}')

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.policy_network(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def update_policy(self, trajectory):
        states, actions, log_probs, rewards, next_states = zip(*trajectory)
        # Compute discounted rewards
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)

        discounted_rewards = torch.tensor(discounted_rewards).to(device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        # Compute policy gradient loss
        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        policy_loss = torch.cat(policy_loss).sum()

        # Update policy network
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()