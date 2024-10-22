import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

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

class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        # Define the two-layer network with 512 units each
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 1)  # Output a single value

        # Initialize weights with uniform distribution based on the input dimensions
        c = np.sqrt(1 / input_dim)
        nn.init.uniform_(self.fc1.weight, -c, c)
        nn.init.uniform_(self.fc1.bias, -c, c)
        nn.init.uniform_(self.fc2.weight, -c, c)
        nn.init.uniform_(self.fc2.bias, -c, c)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # Output the state value directly

class A2CAgent:
    def __init__(self, input_dim, output_dim, gamma=0.95, lr=0.0001):
        self.policy_network = PolicyNetwork(input_dim, output_dim).to(device)
        self.value_network = ValueNetwork(input_dim).to(device)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.policy_network(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def update(self, state, action_log_prob, reward, next_state, done):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)

        # Compute value targets
        value = self.value_network(state)
        next_value = self.value_network(next_state)

        # A2C uses the advantage function explicitly
        target = reward + (1 - done) * self.gamma * next_value
        advantage = (target - value).detach()

        # Compute value loss
        value_loss = F.mse_loss(value, target.detach())

        # Compute policy loss using advantage
        policy_loss = -(action_log_prob * advantage)

        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()