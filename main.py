from unityagents import UnityEnvironment

from MADDPG import MADDPG
import numpy as np
import torch


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
episodes_before_train = 50#100


maddpg = MADDPG(num_agents, state_size, action_size, batch_size, capacity, episodes_before_train)

FloatTensor = torch.cuda.FloatTensor if maddpg.use_cuda else torch.FloatTensor
for i_episode in range(num_episode):
    env_info = env.reset(train_mode=True)[brain_name]
    obs = env_info.vector_observations  # get the current state (for each agent)
    #obs = np.stack(obs)
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
        #obs_ = np.stack(obs_)
        obs_ = torch.from_numpy(obs_).float()
        if t != max_steps - 1:
            next_obs = obs_
        else:
            next_obs = None

        total_reward += reward.sum()
        rr += reward.cpu().numpy()
        maddpg.memory.push(obs.data, action, next_obs, reward)
        obs = next_obs

        c_loss, a_loss = maddpg.update_policy()

    maddpg.episode_done += 1
    print('Episode: %d, reward = %f' % (i_episode, total_reward))
    reward_record.append(total_reward)

    if maddpg.episode_done == maddpg.episodes_before_train:
        print('training now begins...')

env.close()
