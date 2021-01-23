from abc import ABCMeta, abstractmethod

class BaseNoise(metaclass=ABCMeta):
    """Defines base class for noises
    """

    @abstractmethod
    def __init__(self):
        self._step = 0

    @abstractmethod
    def __call__(self, action):
        self._step += 1

    @abstractmethod
    def reset(self):
        self._step = 0