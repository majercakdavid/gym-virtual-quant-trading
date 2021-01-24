import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from torch.utils.tensorboard import SummaryWriter

from gym_virtual_quant_trading.envs.PaperTradingEnv import PaperTradingEnv, PaperTradingEnvConfig
from agents.DQN import DQN
from agents.DDPG import DDPG
from agents.utils.ReplayMemory import ReplayMemory, Transition
from agents.noise.OrnsteinUhlenbeckNoise import OrnsteinUhlenbeckNoise

env_cfg = PaperTradingEnvConfig()
env = PaperTradingEnv(env_cfg)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
GAMMA = 0.6
PRINT_LOSS = 100

init_data = env.reset()
_, in_size = init_data.shape

# Get number of actions from gym action space
n_actions = env.action_space.shape[-1]

net_noise = OrnsteinUhlenbeckNoise(n_actions)
net = DDPG(
    in_size=in_size, 
    action_space_size=n_actions, 
    gamma=GAMMA, 
    actor_lr=1e-6, 
    critic_lr=1e-6, 
    noise=net_noise)

run_id = "{}_{}_{:%Y%m%dT%H%M}".format(
        net.__class__.__name__, env_cfg.DATA_SOURCE.__class__.__name__, datetime.now())
summary_writer = SummaryWriter(log_dir='./.cache/tensorboard', filename_suffix=run_id)


memory = ReplayMemory(10000)
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


num_episodes = 5000
for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()

    # Create state tensor and move it to GPU
    state = torch.tensor(state, dtype=torch.float, device=device)

    for t in count():
        # Select and perform an action
        action = net.select_action(state)
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
            value_loss, policy_loss = net.update_params(memory.sample(BATCH_SIZE))
            portfolio_value = env._get_portfolio_value(env._portfolio)

            episode_loss_value.append([value_loss, policy_loss, env._liquidity, portfolio_value, reward.data.cpu()[0]])

            summary_writer.add_scalar('train/reward', reward, i_episode)
            summary_writer.add_scalar('train/loss/value', value_loss, i_episode)
            summary_writer.add_scalar('train/loss/policy', policy_loss, i_episode)
            summary_writer.add_scalar('train/liquidity', env._liquidity, i_episode)
            summary_writer.add_scalar('train/portfolio_value', portfolio_value, i_episode)

            if t%PRINT_LOSS == 0:
                print(f'Value loss: {value_loss}, Policy loss: {policy_loss}')

        if done:
            break

    summary_writer.flush()
summary_writer.close()

print('Complete')
env.render()
env.close()