import argparse
import numpy as np
import torch
from unityagents import UnityEnvironment

from ddpg_agent import Agent

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', default='/Users/saminda/Udacity/DRLND/Sim/Tennis/Tennis.app',
                    help='Unity environment')
args = parser.parse_args()

print('file_name:', args.file_name)
env = UnityEnvironment(file_name=args.file_name)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
# print('The state for the first agent looks like:', states[0])

agent = Agent(state_size=state_size, action_size=action_size, num_agents=num_agents, random_seed=2)

agent.actor_local.load_state_dict(torch.load('solution_actor.pth', map_location='cpu'))
agent.critic_local.load_state_dict(torch.load('solution_critic.pth', map_location='cpu'))

env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
states = env_info.vector_observations  # get the current state (for each agent)
scores = np.zeros(num_agents)  # initialize the score (for each agent)
while True:
    actions = agent.act(states, add_noise=False)  # select an action (for each agent)
    env_info = env.step(actions)[brain_name]  # send all actions to tne environment
    next_states = env_info.vector_observations  # get next state (for each agent)
    rewards = env_info.rewards  # get reward (for each agent)
    dones = env_info.local_done  # see if episode finished
    scores += env_info.rewards  # update the score (for each agent)
    states = next_states  # roll over states to next time step
    if np.any(dones):  # exit loop if episode finished
        break
print('Maximum score: {}'.format(np.max(scores)))
