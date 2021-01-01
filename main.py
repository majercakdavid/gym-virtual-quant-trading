import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from gym_virtual_quant_trading.envs.PaperTradingEnv import PaperTradingEnv, PaperTradingEnvConfig
from agents.DQN import DQN
from agents.DDPG import DDPG
from agents.ReplayMemory import ReplayMemory, Transition

env_cfg = PaperTradingEnvConfig()
env = PaperTradingEnv(env_cfg)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_data(data):
    transforms = T.Compose([
        T.ToTensor()
    ])
    transforms(data).unsqueeze(0).to(device)

BATCH_SIZE = 32
GAMMA = 0.6
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
PRINT_LOSS = 100

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_data = env.reset()
_, in_size = init_data.shape

# Get number of actions from gym action space
n_actions = env.action_space.shape[-1]

ddpg = DDPG(in_size=in_size, action_space_size=n_actions, gamma=GAMMA, actor_lr=1e-6, critic_lr=1e-6)

# policy_net = DQN(input_dim=(y, x), output_dim=n_actions).to(device)
# target_net = DQN(input_dim=(y, x), output_dim=n_actions).to(device)
# target_net.load_state_dict(policy_net.state_dict())
# target_net.eval()

# optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state, policy_net):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            policy_action = policy_net(state)

            return policy_action.view(-1)
    else:
        return (torch.rand(3)*2-1).to(device)

episode_loss_value = []

def plot_episode():
    loss_value = torch.tensor(episode_loss_value, dtype=torch.float)
    
    plt.subplot(2, 2, 1)
    plt.title('Training')
    plt.xlabel('Time')
    plt.ylabel('Value loss')
    plt.plot(loss_value[:, 0].numpy(), label='Value loss')

    plt.subplot(2, 2, 2)
    plt.title('Training')
    plt.xlabel('Time')
    plt.ylabel('Policy loss')
    plt.plot(loss_value[:, 1].numpy(), label='Policy loss')

    plt.subplot(2, 2, 3)
    plt.title('Training')
    plt.xlabel('Time')
    plt.ylabel('Liquidity value')
    plt.plot(loss_value[:, 2].numpy(), label='Liquidity')

    plt.subplot(2, 2, 4)
    plt.title('Training')
    plt.xlabel('Time')
    plt.ylabel('Portfolio value')
    plt.plot(loss_value[:, 3].numpy(), label='Portfolio')

    plt.pause(0.1)
    plt.show()

    plt.plot()
    plt.title('Training')
    plt.xlabel('Time')
    plt.ylabel('Reward')
    plt.plot(loss_value[:, 4].numpy())

    plt.pause(0.1)
    plt.show()

    # plt.pause(0.001)  # pause a bit so that plots are updated


num_episodes = 50
for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()

    # Create state tensor and move it to GPU
    state = torch.tensor(state, dtype=torch.float, device=device)

    for t in count():
        # Select and perform an action
        action = select_action(state, ddpg.actor)
        next_state, reward, done, _ = env.step(action.data.cpu().numpy())
        
        mask = torch.tensor([done], device=device)
        reward = torch.tensor([reward], device=device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=device)

        # Store the transition in memory
        memory.push(state, action.view(1,-1), next_state, reward, mask)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        if len(memory) > BATCH_SIZE:
            value_loss, policy_loss = ddpg.update_params(memory.sample(BATCH_SIZE))
            episode_loss_value.append([value_loss, policy_loss, env._liquidity, env._get_portfolio_value(env._portfolio), reward.data.cpu()[0]])
            if t%PRINT_LOSS == 0:
                print(f'Value loss: {value_loss}, Policy loss: {policy_loss}')

        if done:
            plot_episode()
            break

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()