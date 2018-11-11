from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment

from MADDPG import MADDPG

reward_record = []

np.random.seed(1234)
torch.manual_seed(1234)

#env = UnityEnvironment(file_name="/home/ubuntu/Tennis_Linux_NoVis/Tennis.x86_64")
env = UnityEnvironment(file_name="/Users/saminda/Udacity/DRLND/Sim/Tennis/Tennis.app")

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
batch_size = 1000

num_episode = 3000
max_steps = 1000
save_interval = 100

scores_deque = deque(maxlen=100)
scores_list = []
avg_list = []
tot_list = []

maddpg = MADDPG(num_agents, state_size, action_size, batch_size, capacity)

FloatTensor = torch.cuda.FloatTensor if maddpg.use_cuda else torch.FloatTensor
for i_episode in range(num_episode):
    maddpg.reset()
    env_info = env.reset(train_mode=True)[brain_name]
    obs = env_info.vector_observations  # get the current state (for each agent)
    obs = torch.from_numpy(obs).float()
    total_reward = 0.0
    rr = np.zeros((num_agents,))
    episode_done = False
    for t in range(max_steps):
        obs = obs.type(FloatTensor)
        action = maddpg.select_action(obs).data.cpu()
        env_info = env.step(action.numpy())[brain_name]
        next_obs, reward, done, = env_info.vector_observations, env_info.rewards, env_info.local_done

        if np.any(done):
            episode_done = True

        reward = torch.FloatTensor(reward).type(FloatTensor)
        done = torch.FloatTensor(done).type(FloatTensor)
        next_obs = torch.from_numpy(next_obs).float()

        total_reward += reward.sum()
        rr += reward.cpu().numpy()
        maddpg.memory.push(obs, action, next_obs, reward, done)
        obs = next_obs

        if maddpg.steps_done % 100 == 0:
            for _ in range(10):
                c_loss, a_loss = maddpg.update_policy()

        if episode_done:
            break

    maddpg.episode_done += 1
    reward_record.append(total_reward)

    if i_episode % save_interval == 0:
        print()
        save_dict_list = []
        for i in range(num_agents):
            save_dict = {'actor_params': maddpg.actors[i].state_dict(),
                         'actor_optim_params': maddpg.actor_optimizer[i].state_dict(),
                         'critic_params': maddpg.critics[i].state_dict(),
                         'critic_optim_params': maddpg.critic_optimizer[i].state_dict()}
            save_dict_list.append(save_dict)
        torch.save(save_dict_list, 'model-{}.bin'.format(maddpg.__class__.__name__))

    score = np.max(rr)
    scores_list.append(score)
    scores_deque.append(score)

    avg = np.average(scores_deque)
    avg_list.append(avg)
    tot_list.append(total_reward)

    print(f"\rEpisode: {i_episode:4d}  Average Score: {avg:.4f}", end="")

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores_list) + 1), scores_list)
plt.plot(np.arange(1, len(avg_list) + 1), avg_list)
plt.plot(np.arange(1, len(tot_list) + 1), tot_list)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('scores.png')

env.close()
