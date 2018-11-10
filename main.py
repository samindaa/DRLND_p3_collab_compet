from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment

from MADDPG import MADDPG

reward_record = []

np.random.seed(1234)
torch.manual_seed(1234)

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

num_episode = 20000
max_steps = 1000
episodes_before_train = 100
save_interval = 10

scores_deque = deque(maxlen=100)
scores_list = []
avg_list = []

maddpg = MADDPG(num_agents, state_size, action_size, batch_size, capacity, episodes_before_train)

FloatTensor = torch.cuda.FloatTensor if maddpg.use_cuda else torch.FloatTensor
for i_episode in range(num_episode):
    env_info = env.reset(train_mode=True)[brain_name]
    obs = env_info.vector_observations  # get the current state (for each agent)
    # obs = np.stack(obs)
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs).float()
    total_reward = 0.0
    rr = np.zeros((num_agents,))
    for t in range(max_steps):

        obs = obs.type(FloatTensor)
        action = maddpg.select_action(obs).data.cpu()
        env_info = env.step(action.numpy())[brain_name]

        obs_, reward, done, = env_info.vector_observations, env_info.rewards, env_info.local_done

        reward = torch.FloatTensor(reward).type(FloatTensor)
        # obs_ = np.stack(obs_)
        obs_ = torch.from_numpy(obs_).float()
        if t != max_steps - 1 or not np.any(done):
            next_obs = obs_
        else:
            next_obs = None

        total_reward += reward.sum()
        rr += reward.cpu().numpy()
        maddpg.memory.push(obs.data, action, next_obs, reward)
        obs = next_obs

        c_loss, a_loss = maddpg.update_policy()

        if np.any(done):
            break

    maddpg.episode_done += 1
    reward_record.append(total_reward)

    if maddpg.episode_done < maddpg.episodes_before_train:
        print('Episode: %d, reward = %f' % (i_episode, total_reward))

    if maddpg.episode_done == maddpg.episodes_before_train:
        print('training now begins...')

    if maddpg.episode_done > maddpg.episodes_before_train and i_episode % save_interval == 0:
        save_dict_list = []
        for i in range(num_agents):
            save_dict = {'actor_params': maddpg.actors[i].state_dict(),
                         'actor_optim_params': maddpg.actor_optimizer[i].state_dict(),
                         'critic_params': maddpg.critics[i].state_dict(),
                         'critic_optim_params': maddpg.critic_optimizer[i].state_dict()}
            save_dict_list.append(save_dict)
        torch.save(save_dict_list, 'model-{}.bin'.format(maddpg.__class__.__name__))

    if maddpg.episode_done > maddpg.episodes_before_train:
        score = np.max(rr)
        scores_list.append(score)
        scores_deque.append(score)

        avg = np.average(scores_deque)
        avg_list.append(avg)

        print(f"\rEpisode: {i_episode:4d}   Episode Score: {score:.2f}   Average Score: {avg:.4f}", end="")

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores_list) + 1), scores_list)
plt.plot(np.arange(1, len(avg_list) + 1), avg_list)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('scores.png')

env.close()
