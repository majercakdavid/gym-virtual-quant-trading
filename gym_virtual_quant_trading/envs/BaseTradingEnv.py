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
        self._liquidity         = 0                                     # type: float

    @abstractmethod
    def step(self, action):
        """Perform trade operation on the market

        Args:
            action (List[BaseTradingEnvActions]/List[tuple(BaseTradingEnvActions, int)]): Type of the trade and amount of shares to trade (default: 1)

        Returns:
            float: reward for executed action
        """
        if(type(action) != list ):
            raise ValueError("Action must be a list")

        if(type(action[0]) != tuple ):
            # TODO: Make default value for number of shares to exercise configurable
            action = [(a, 1) for a in action] # Assign amount to each action

        if not all([isinstance(a[0], BaseTradingEnvActions) for a in action]):
            try:
                action = [(BaseTradingEnvActions(a[0]), a[1]) for a in action]
            except:
                raise ValueError("Action must contain either BaseTradingEnvActions instances or values that are convertable")


        observation = next(self._config.DATA_SOURCE)    # type: data.DefaultMarketData
        reward      = 0                                 # type: int initial reward              
        
        sells = {
            instrument.symbol: instrument.open 
            for i, instrument in enumerate(observation.entries.values()) 
            if action[i][0] == BaseTradingEnvActions.SELL
        }

        buys = {
            instrument.symbol: instrument.open 
            for i, instrument in enumerate(observation.entries.values()) 
            if action[i][0] == BaseTradingEnvActions.BUY
        }

        # Naive reward:
        #   Do not sell and buy same stock at one time
        #   Do not sell more than you own
        #   Do not buy more than you can afford
        reward -= len(set(sells.keys()).intersection(buys.keys))
        reward -= len([max(0, self._portfolio[sell[0]] - sell[1]) for sell in sells])
        reward -= 0 if (self._liquidity - sum(buys.values())) > 0 else len(buys)

        return reward


        
    @abstractmethod
    def reset(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        self._liquidity = self._config.INITIAL_BALANCE
        data = next(self._config.DATA_SOURCE)   # type: data.DefaultMarketData
        return self._create_observation_space(data)

    def _create_observation_space(self, data):
        """Performs an observation of environment

        Returns:
            np.array: array containing all OBSERVED_SYMBOL_PROPS for every symbol at current time step
        """
        selected_props = [
            value.__getattribute__(prop) 
            for prop in self._config.OBSERVED_SYMBOL_PROPS
            for value in data.entries.values()
        ]
        selected_props = np.array([self._net_worth, *selected_props])

        return selected_props