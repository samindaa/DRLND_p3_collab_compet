import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(dim_observation, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, dim_action)
        self.bn3 = nn.BatchNorm1d(dim_action)
        self.reset_parameters()

    # action output between -1 and 1
    def forward(self, obs):
        x = self.bn1(F.relu(self.fc1(obs)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = F.tanh(self.bn3(self.fc3(x)))
        return x

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)


class Critic(nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = dim_observation * n_agent
        act_dim = self.dim_action * n_agent

        self.bn0 = nn.BatchNorm1d(obs_dim)
        self.fc1 = nn.Linear(obs_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128 + act_dim, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)
        self.reset_parameters()

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        xs = F.leaky_relu(self.bn1(self.fc1(self.bn0(obs))))
        x = torch.cat([xs, acts], 1)
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
