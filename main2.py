from collections import deque
from itertools import count

import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment

from ddpg_agent import Agent

# %matplotlib inline

env = UnityEnvironment(file_name="/home/ubuntu/Tennis_Linux_NoVis/Tennis.x86_64")
# env = UnityEnvironment(file_name="/Users/saminda/Udacity/DRLND/Sim/Tennis/Tennis.app")

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


def save_model():
    print("Model Save...")
    torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
    torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')


def ddpg(n_episodes=3200, print_every=10, save_every=100):
    avg_solved = 0
    scores_deque = deque(maxlen=100)
    scores_global = []
    avg_global = []

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state (for each agent)
        scores = np.zeros(num_agents)  # initialize the score (for each agent)
        agent.reset()

        # timestep = time.time()
        for t in count():
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]  # send all actions to tne environment
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states  # roll over states to next time step
            scores += rewards  # update the score (for each agent)
            if np.any(dones):  # exit loop if episode finished
                break

        score = np.max(scores)
        scores_deque.append(score)
        score_average = np.mean(scores_deque)
        scores_global.append(score)
        avg_global.append(score_average)

        if i_episode % save_every == 0:
            save_model()

        if i_episode % print_every == 0:
            print('\rEpisode {}, Average Score: {:.2f}, Max: {:.2f}, Min: {:.2f}' \
                  .format(i_episode, score_average, np.max(scores), np.min(scores)), end="\n")

        if score_average >= 0.5:
            if avg_solved == 0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, score_average))
            if score_average > avg_solved:
                avg_solved = score_average
                print('\nSaving: {:d} Average Score: {:.2f}'.format(i_episode, avg_solved))
                save_model()

    return scores_global, avg_global


scores, avgs = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.plot(np.arange(1, len(avgs) + 1), avgs)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('scores.png')
