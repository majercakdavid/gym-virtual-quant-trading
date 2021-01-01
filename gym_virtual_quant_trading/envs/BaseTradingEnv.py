from enum import Enum
from copy import deepcopy
from abc import ABCMeta, abstractmethod

import numpy as np
import gym
from gym.spaces import Discrete, Box, Dict

import gym_virtual_quant_trading.data as data

class BaseTradingEnvActions(Enum):
    """Defines action space of BaseTradingEnv"""
    SELL = 0
    FLAT = 1
    BUY  = 2

class BaseTradingEnvConfig():
    INITIAL_BALANCE         = 100000.0                          # type: float Defines initial amount available to invest
    LEVERAGE                = 1                                 # type: float Defines available leverage on financial instruments
    DATA_SOURCE             = data.DefaultMarketDataSource()    # type: data.BaseMarketDataSource Defines data source for market data
    OBSERVED_SYMBOLS        = ['AAPL', 'AMZN', 'MSFT']          # type: List[str] symbols to watch
    OBSERVED_SYMBOL_PROPS   = [                                 # type: List[str] properties of symbols to use  in observation space
        'open', 'high', 'low', 'close', 'volume'
        ]
    TRADE_MAX               = 1000                              # type int Defines max amount of instruments in one trade
    TRADE_MAX_AMOUNT        = False                             # type: bool Defines whether to convert continuous action space to:
                                                                #   True:   amount of instruments
                                                                #   False:  value of the trade

class BaseTradingEnv(gym.Env, metaclass=ABCMeta):
    """Base OpenAI Gym Trading Environment

    TODO: Add description
    Args:
        gym ([type]): [description]
    """

    @abstractmethod
    def __init__(self, baseTradingEnvConfig: BaseTradingEnvConfig, discrete_action_space=False):
        """Initialized the BaseTradingEnv class

        Args:
            baseTradingEnvConfig (BaseTradingEnvConfig): configuration

        Raises:
            ValueError: baseTradingEnvConfig must be of type BaseTradingEnvConfig
        """
        super(BaseTradingEnv, self).__init__()

        if not isinstance(baseTradingEnvConfig, BaseTradingEnvConfig):
            raise ValueError("baseTradingEnvConfig must be derived from BaseTradingEnvConfig")

        # Action space defines actions we can do in every step, for trading these can be either:
        if discrete_action_space:
            # Discrete: buy, sell, flat
            self.action_space       = Discrete(                 # type: Discrete
                len(BaseTradingEnvActions)*len(baseTradingEnvConfig.OBSERVED_SYMBOLS)
            )  
        else: 
            # Continuous: buy(action>0), sell(action<0), flat(action==0)
            # Additionally the higher the absolute value of an action the more
            # instruments are bought/sold
            self.action_space       = Box(                      # type: Box
                low=-1, high=np.inf,
                shape=(len(baseTradingEnvConfig.OBSERVED_SYMBOLS),)
            )
        
        # The information about financial instruments in current state containing:
        #   Liquidity
        #   Configured observed properties for every symbol
        #   Portfolio size and value for every symbol
        self.observation_space = Box(low=0, high=np.inf,        # type: Box
            shape=(
                1 + len(baseTradingEnvConfig.OBSERVED_SYMBOLS)*(len(baseTradingEnvConfig.OBSERVED_SYMBOL_PROPS)+1),
            )
        )
        
        self._config            = baseTradingEnvConfig      # type: BaseTradingEnvConfig
        self._portfolio         = {                         # type: Dict[str, int] 
            symbol: [] for symbol in baseTradingEnvConfig.OBSERVED_SYMBOLS
        }
        self._liquidity         = 0                         # type: float
        self._last_obs          = None                      # type: Dict

    @abstractmethod
    def step(self, action):
        """Perform trade operation on the market

        Args:
            action (List[BaseTradingEnvActions]/List[tuple(BaseTradingEnvActions, int)]): Type of the trade and amount of shares to trade (default: 1)

        Returns:
            float   : reward for executed action
            np.array: array containing all OBSERVED_SYMBOL_PROPS for every symbol at current time step
        """
        if(type(action) != list and type(action) != np.ndarray):
            raise ValueError("Action must be a list")

        if not all(a<=1 and a>=-1 for a in action) or\
            not len(action) == len(self._config.OBSERVED_SYMBOLS):
            raise ValueError("Action must be a list of values between -1 and 1 for every observed financial instrument")
        
        action = action*self._config.TRADE_MAX
        observation = next(self._config.DATA_SOURCE)    # type: data.DefaultMarketData   
    
        try:
            action = [
                (BaseTradingEnvActions.BUY if a >= .5 else (BaseTradingEnvActions.SELL if a <= -.5 else BaseTradingEnvActions.FLAT), abs(a))
                for a in action
            ]
        except:
            raise ValueError("Action must contain either BaseTradingEnvActions instances or values that are convertable")    
    
        if observation is None:
            observation     = self._last_obs
            done            = True
        else:
            self._last_obs  = observation
            done            = False 

        sells = {
            symbol: int(action[i][1]) if self._config.TRADE_MAX_AMOUNT else\
                int(action[i][1]//observation.entries[symbol].open)
            for i, symbol in enumerate(self._config.OBSERVED_SYMBOLS)
            if action[i][0] == BaseTradingEnvActions.SELL
        }

        buys = {
            symbol: int(action[i][1]) if self._config.TRADE_MAX_AMOUNT else\
                int(action[i][1]//observation.entries[symbol].open)
            for i, symbol in enumerate(self._config.OBSERVED_SYMBOLS)
            if action[i][0] == BaseTradingEnvActions.BUY
        }

        # Naive reward:
        #   Do not sell and buy same stock at one time
        #   Do not sell more than you own
        #   Do not buy more than you can afford (but encourage the system to buy more)
        reward_bs   = -len(set(sells.keys()).intersection(buys.keys()))                     # type: int
        
        reward_sell = -len([                                                                # type: int 
            max(
                0, sum([p[1] for p in self._portfolio[sell[0]]]) - sell[1]
            ) for sell in sells.items()
        ])
        reward_buy = 0 if (self._liquidity - sum([                                  # type: int
            observation.entries[symbol].open*amount for symbol, amount in buys.items()
        ])) > 0 else -len(buys)
                                          
        new_portfolio = deepcopy(self._portfolio)
        liquidity_diff = 0
        trade_gains = 0
        if reward_sell == 0:
            for symbol, amount in sells.items():
                to_sell = amount
                liquidity_diff += observation.entries[symbol].open*amount
                
                while to_sell > 0:
                    if new_portfolio[symbol][0][1] <= to_sell:
                        trade_gains += new_portfolio[symbol][0][1]*(
                            observation.entries[symbol].open - new_portfolio[symbol][0][2]
                        ) 

                        to_sell -= new_portfolio[symbol][0][1]
                        new_portfolio[symbol] = new_portfolio[symbol][1:]
                    else:
                        trade_gains += to_sell*(
                            observation.entries[symbol].open - new_portfolio[symbol][0][2]
                        )

                        to_sell = 0 
                        new_portfolio[symbol][0][1] -= to_sell
        reward_sell = len(sells)
        
        if reward_buy >= 0:
            for symbol, amount in buys.items():
                liquidity_diff -= observation.entries[symbol].open*amount
                new_portfolio[symbol].append([observation.time, amount, observation.entries[symbol].open])

        self._liquidity += liquidity_diff 
        self._portfolio = new_portfolio

        debug   = {}                                                                        # type: Dict
        reward  = reward_bs + reward_sell + reward_buy                                      # type: int initial reward
        reward += trade_gains

        return self._create_observation_space(observation), reward, done, debug
        
    @abstractmethod
    def reset(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        self._liquidity = self._config.INITIAL_BALANCE
        self._portfolio = {
            symbol: [] for symbol in self._config.OBSERVED_SYMBOLS
        }
        self._config.DATA_SOURCE.reset()
        data = next(self._config.DATA_SOURCE)   # type: data.DefaultMarketData
        return self._create_observation_space(data)

    def _get_portfolio_value(self, portfolio):
        return sum([trade[1]*trade[2] for symbol_trades in portfolio.values() for trade in symbol_trades])

    def _create_observation_space(self, data):
        """Performs an observation of environment

        Returns:
            np.array: array containing all OBSERVED_SYMBOL_PROPS for every symbol at current time step
        """
        if data is None:
            return data

        selected_props = [
            data.entries[name].__getattribute__(prop) 
            for name in self._config.OBSERVED_SYMBOLS
            for prop in self._config.OBSERVED_SYMBOL_PROPS
            if name in self._config.DATA_SOURCE.symbols()
        ]
        portfolio = [
            [sum([p[1] for p in self._portfolio.get(symbol, [0, 0, 0])]),
            sum([p[1]*p[2] for p in self._portfolio.get(symbol, [0, 0, 0])])]
            for symbol in self._config.OBSERVED_SYMBOLS
            if symbol in self._config.DATA_SOURCE.symbols()
        ]
        portfolio = [item for p in portfolio for item in p]

        selected_props = np.array([self._liquidity, *portfolio, *selected_props])

        return selected_props.reshape(1, -1)