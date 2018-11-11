from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from memory import ReplayMemory, Experience
from networks import MLPNetwork
from noise import OUNoise


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class MADDPG:
    def __init__(self, n_agents, dim_obs, dim_act, batch_size, capacity):
        self.actors = [MLPNetwork(dim_obs, dim_act,
                                 hidden_dim=64,
                                 constrain_out=True) for i in range(n_agents)]
        self.critics = [MLPNetwork(n_agents * (dim_obs + dim_act), 1,
                                 hidden_dim=64,
                                 constrain_out=False) for i in range(n_agents)]
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.use_cuda = torch.cuda.is_available()

        self.GAMMA = 0.99
        self.tau = 1e-3

        self.action_noise = [OUNoise(self.n_actions, mu=np.zeros(self.n_actions)) for i in range(n_agents)]
        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=0.001) for x in self.critics] #, weight_decay=0.0001
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.0001) for x in self.actors]

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

        self.steps_done = 0
        self.episode_done = 0

    def update_policy(self):
        # do not train until exploration is enough
        if len(self.memory) < self.batch_size:
            return None, None

        #ByteTensor = torch.cuda.ByteTensor if self.use_cuda else torch.ByteTensor
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        c_loss = []
        a_loss = []
        for agent in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))

            # state_batch: batch_size x n_agents x dim_obs
            state_batch = torch.stack(batch.states).type(FloatTensor)
            action_batch = torch.stack(batch.actions).type(FloatTensor)
            # : (batch_size_non_final) x n_agents x dim_obs
            next_states = torch.stack(batch.next_states).type(FloatTensor)
            reward_batch = torch.stack(batch.rewards).type(FloatTensor)
            done_batch = torch.stack(batch.dones).type(FloatTensor)

            # for current agent
            whole_state = state_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)
            self.critic_optimizer[agent].zero_grad()
            current_Q = self.critics[agent](torch.cat([whole_state, whole_action], dim=1))

            next_actions = [
                self.actors_target[i](next_states[:,
                                      i,
                                      :]) for i in range(
                    self.n_agents)]
            next_actions = torch.stack(next_actions)
            next_actions = (
                next_actions.transpose(0,
                                                 1).contiguous())

            target_Q = self.critics_target[agent](
                torch.cat(
                    [next_states.view(-1, self.n_agents * self.n_states),
                     next_actions.view(-1, self.n_agents * self.n_actions)], dim=1)
            ).squeeze()

            target_Q = ((1.0 - (done_batch[:, agent].unsqueeze(1))) * target_Q.unsqueeze(1) * self.GAMMA) + (
                reward_batch[:, agent].unsqueeze(1))

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent].parameters(), 0.5)
            self.critic_optimizer[agent].step()

            self.actor_optimizer[agent].zero_grad()
            state_i = state_batch[:, agent, :]
            action_i = self.actors[agent](state_i)
            ac = action_batch.clone()
            ac[:, agent, :] = action_i
            whole_action = ac.view(self.batch_size, -1)
            actor_loss = -self.critics[agent](torch.cat([whole_state, whole_action], dim=1))
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent].parameters(), 0.5)
            self.actor_optimizer[agent].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        #if self.steps_done % 100 == 0 and self.steps_done > 0:
        for i in range(self.n_agents):
            soft_update(self.critics_target[i], self.critics[i], self.tau)
            soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss

    def select_action(self, state_batch):
        # state_batch: n_agents x state_dim
        actions = torch.zeros(
            self.n_agents,
            self.n_actions)
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        for i in range(self.n_agents):
            sb = state_batch[i, :].detach()
            self.actors[i].eval()
            with torch.no_grad():
                act = self.actors[i](sb.unsqueeze(0)).squeeze()
            self.actors[i].train()
            act += torch.from_numpy(self.action_noise[i].noise()).type(FloatTensor)
            act = torch.clamp(act, -1.0, 1.0)
            actions[i, :] = act
        self.steps_done += 1

        return actions

    def reset(self):
        for i in range(self.n_agents):
            self.action_noise[i].reset()
