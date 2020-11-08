from enum import Enum
from abc import ABCMeta, abstractmethod

import numpy as np
import gym
from gym.spaces import Discrete, Box, Dict

import gym_virtual_quant_trading.data as data

class BaseTradingEnvActions(Enum):
    """Defines action space of BaseTradingEnv"""
    SELL = -1
    FLAT =  0
    BUY  =  1

class BaseTradingEnvConfig():
    INITIAL_BALANCE         = 10000.0                           # type: float Defines initial amount available to invest
    LEVERAGE                = 1                                 # type: float Defines available leverage on financial instruments
    DATA_SOURCE             = data.DefaultMarketDataSource()    # type: data.BaseMarketDataSource Defines data source for market data
    OBSERVED_SYMBOL_PROPS   = [                                 # type: List[str] properties of symbols to use  in observation space
        'open', 'high', 'low', 'close', 'volume'
        ]
    OBSERVED_STEPS          = 5                                 # type: int number of previous timesteps to include in observation space

class BaseTradingEnv(gym.Env, metaclass=ABCMeta):
    """Base OpenAI Gym Trading Environment

    TODO: Add description
    Args:
        gym ([type]): [description]
    """

    @abstractmethod
    def __init__(self, baseTradingEnvConfig: BaseTradingEnvConfig):
        """Initialized the BaseTradingEnv class

        Args:
            baseTradingEnvConfig (BaseTradingEnvConfig): configuration

        Raises:
            ValueError: baseTradingEnvConfig must be of type BaseTradingEnvConfig
        """
        super(BaseTradingEnv, self).__init__()

        if not isinstance(baseTradingEnvConfig, BaseTradingEnvConfig):
            raise ValueError("baseTradingEnvConfig must be derived from BaseTradingEnvConfig")

        # Action space defines actions we can do in every step, for trading these are: buy, sell, flat
        self.action_space       = Discrete(len(BaseTradingEnvActions))  # type: Discrete

        # The prices of financial instruments in current state
        self.observation_space  = Dict({                                # type: Dict
            symbol: Box(low=0, high=np.inf, shape=(
                baseTradingEnvConfig.OBSERVED_STEPS, 
                len(baseTradingEnvConfig.OBSERVED_SYMBOL_PROPS)
                )) 
            for symbol in baseTradingEnvConfig.DATA_SOURCE.symbols()
        })
        
        self._config            = baseTradingEnvConfig                  # type: BaseTradingEnvConfig
        self._portfolio         = {}                                    # type: Dict[str, int] 
        self._net_worth         = 0                                     # type: float

    @abstractmethod
    def step(self, symbol, action, amount):
        """Perform trade operation on the market

        Args:
            symbol (str): Symbol to trade
            action (BaseTradingEnvActions): Type of the trade
            amount (float): Amount to trade

        Returns:
            float: reward for executed action
        """
        pass
        
    @abstractmethod
    def reset(self):
        self._net_worth = self._config.INITIAL_BALANCE
