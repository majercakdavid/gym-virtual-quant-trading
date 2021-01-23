import torch
import numpy as np
from agents.noise.BaseNoise import BaseNoise

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OrnsteinUhlenbeckNoise(BaseNoise):
    """Generates Ornstein-Uhlenbeck noise (https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process)
    """
    def __init__(self, action_space_dim, mu = 0, sigma=0.4, theta=.01, scale=0.1):
        super(OrnsteinUhlenbeckNoise, self).__init__()

        self.theta = theta
        self.mu = mu*np.ones(action_space_dim)
        self.sigma = sigma
        self.scale = scale
        self.reset()

    def __call__(self, action):
        super().__call__(action)
        x = self.x_prev + self.theta * (self.mu - self.x_prev) + \
            self.sigma * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return action + torch.Tensor(x * self.scale).to(device)

    def reset(self):
        super().reset()
        self.x_prev = np.zeros_like(self.mu)