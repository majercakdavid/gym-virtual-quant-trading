import torch

from abc import ABCMeta, abstractmethod
from torch import nn

class BaseAgent(nn.Module, metaclass=ABCMeta):

    @property
    @abstractmethod
    def action_policy(self):
        return None

    @abstractmethod
    def __init__(self, noise=None):
        super(BaseAgent, self).__init__()
        self._noise = noise

    @abstractmethod
    def save_model(self):
        pass

    def select_action(self, state):
        with torch.no_grad():
            action = self.action_policy(state).view(-1)

            if self._noise is not None:
                action = self._noise(action)

            return torch.clamp(action, min=-1, max=1)
