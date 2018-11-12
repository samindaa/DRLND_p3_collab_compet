[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Project 3: Collaboration and Competition
## Saminda Abeyruwan

### Introduction

For this project, we have successfully trained and evaluated an agent to solve the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01, otherwise it receives a reward of 0. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation, and the observation vector has 24 features.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, our agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). 

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

We have used [DDGP](https://arxiv.org/abs/1509.02971) algorithm to developed the agent, and we have solved the problem within __1655__ episodes. The weights are saved in [solution_actor.pth](solution_actor.pth) and [solution_critic.pth](solution_critic.pth). 


### Report

The [Report.md](Report.md) contains the detail description of the methodology used in the development of the agent.  

### Training

	python3 train.py --file_name=path/to/Tennis.app

The agent was trained on an AWS p2.xlarge instance. 	

### Testing

	python3 eval.py --file_name=path/to/Tennis.app
	
Please provide the path to the simulator binary or app in the __file\_name__	argument.


### Installing Dependencies

In order to train and test the agent, we need to install and setup the dependencies as follows:

1. git clone [https://github.com/udacity/deep-reinforcement-learning.git](https://github.com/udacity/deep-reinforcement-learning.git)
2. Follow the instructions in the _Dependencies_ section to setup the _drlnd_ in an Anaconda3 environment with Python 3.6.
3. Activate the _drlnd_ environment.
4.  _cd deep-reinforcement-learning/python_ and install the dependencies with the command _pip install ._ (please note  that the dot (__.__) is included. The _requirements.txt_ file is available in _deep-reinforcement-learning/python_).  
5. git clone [https://github.com/samindaa/DRLND\_p3\_collab\_compet.git](https://github.com/samindaa/DRLND_p3_collab_compet.git)
6. _cd DRLND\_p2\_continuous\_control_
7. Download the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) Unity environment. 
8. Follow the instructions in the _Training_ and _Testing_ sections in this document to train and test the agent.  