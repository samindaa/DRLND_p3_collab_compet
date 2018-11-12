from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment

from MADDPG import MADDPG
from memory import ReplayMemory

reward_record = []

np.random.seed(2)
torch.manual_seed(2)

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

capacity = 1000000
batch_size = 1024

num_episode = 6000
max_steps = 1000
save_interval = 100

scores_deque = deque(maxlen=100)
scores_list = []
avg_list = []
tot_list = []

which_agent = 0
avg_solved = 0

pos_examples = ReplayMemory(int(capacity/2))

maddpg = MADDPG(num_agents, state_size, action_size, batch_size, capacity)

for i_episode in range(num_episode):
    maddpg.reset()
    env_info = env.reset(train_mode=True)[brain_name]
    obs = env_info.vector_observations  # get the current state (for each agent)
    obs = torch.from_numpy(obs).float()
    total_reward = 0.0
    rr = np.zeros((num_agents,))
    episode_done = False
    for t in range(max_steps):
        obs = obs.type(maddpg.FloatTensor)
        action = maddpg.select_action(obs).data.cpu()
        env_info = env.step(action.numpy())[brain_name]
        next_obs = env_info.vector_observations
        reward = env_info.rewards
        done = env_info.local_done

        if np.any(done):
            episode_done = True

        reward = torch.FloatTensor(reward).type(maddpg.FloatTensor)
        done = torch.FloatTensor(done).type(maddpg.FloatTensor)
        next_obs = torch.from_numpy(next_obs).float()

        total_reward += reward.sum()
        rr += reward.cpu().numpy()
        maddpg.memory.push(obs, action, next_obs, reward, done)
        if reward.sum() > 0:
            pos_examples.push(obs, action, next_obs, reward, done)
        obs = next_obs

        if maddpg.steps_done % 20 == 0:
            agent = which_agent % maddpg.n_agents
            which_agent += 1
            for _ in range(10):
                c_loss, a_loss = maddpg.update_policy(agent)

        if episode_done:
            break

    maddpg.episode_done += 1
    reward_record.append(total_reward)

    score = np.max(rr)
    scores_list.append(score)
    scores_deque.append(score)

    avg = np.average(scores_deque)
    avg_list.append(avg)
    tot_list.append(total_reward)

    print(f"\rEpisode: {i_episode:4d}  Average Score: {avg:.4f} Pos: {len(pos_examples)}", end="")

    if i_episode % save_interval == 0 and i_episode > 0:
        print()
        maddpg.save('model')

    if avg >= 0.5:
        if avg_solved == 0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, avg))
        if avg > avg_solved:
            avg_solved = avg
            print('\nSaving: {:d} Average Score: {:.2f}'.format(i_episode, avg))
            maddpg.save('model-solution')

    if  i_episode % 100 and len(pos_examples) > 16:
        samples = pos_examples.sample(16)
        for sample in samples:
            maddpg.memory.push(sample.states, sample.actions, sample.next_states, sample.rewards, sample.dones)


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores_list) + 1), scores_list)
plt.plot(np.arange(1, len(avg_list) + 1), avg_list)
plt.plot(np.arange(1, len(tot_list) + 1), tot_list)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('scores.png')

env.close()
