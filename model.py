import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Critic(nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = dim_observation * n_agent
        act_dim = self.dim_action * n_agent

        self.FC1 = nn.Linear(obs_dim, 1024)
        self.FC2 = nn.Linear(1024 + act_dim, 512)
        self.FC3 = nn.Linear(512, 300)
        self.FC4 = nn.Linear(300, 1)
        self.reset_parameters()

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        result = F.relu(self.FC1(obs))
        combined = torch.cat([result, acts], 1)
        result = F.relu(self.FC2(combined))
        return self.FC4(F.relu(self.FC3(result)))

    def reset_parameters(self):
        self.FC1.weight.data.uniform_(*hidden_init(self.FC1))
        self.FC2.weight.data.uniform_(*hidden_init(self.FC2))
        self.FC3.weight.data.uniform_(*hidden_init(self.FC3))
        self.FC4.weight.data.uniform_(-3e-3, 3e-3)


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()
        self.FC1 = nn.Linear(dim_observation, 500)
        self.FC2 = nn.Linear(500, 128)
        self.FC3 = nn.Linear(128, dim_action)
        self.reset_parameters()

    # action output between -2 and 2
    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = F.tanh(self.FC3(result))
        return result

    def reset_parameters(self):
        self.FC1.weight.data.uniform_(*hidden_init(self.FC1))
        self.FC2.weight.data.uniform_(*hidden_init(self.FC2))
        self.FC3.weight.data.uniform_(-3e-3, 3e-3)
