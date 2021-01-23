import math
import torch 

from random import random
from agents.noise.BaseNoise import BaseNoise

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EpsGreedyNoise(BaseNoise):

    def __init__(self, action_space_dim, eps_start=.9, eps_end=.05, eps_decay=200):
        self._eps_start = eps_start
        self._eps_end = eps_end
        self._eps_decay = eps_decay
        self._action_space_dim = action_space_dim

    def __call__(self, action):
        eps = random.random()
        eps_threshold = self._eps_end + (self._eps_start - self._eps_end) * \
            math.exp(-1. * self._step / self._eps_decay)
        if eps > eps_threshold:
            return action
        else:
            return (torch.rand(self._action_space_dim)*2-1).to(device)