# Source: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import random
from functools import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self._capacity  = capacity  # type: int
        self._memory    = []        # type: List[Transition]
        self._position  = 0         # type: int

    def push(self, *args):
        if len(self._memory) < self._capacity:
            self._memory.append(None)

        self._memory[self._position] = Transition(*args)
        self._position = (self._position + 1) % self._capacity

    def sample(self, batch_size):
        return random.sample(self._memory, batch_size)

    def __len__(self):
        return len(self._memory)