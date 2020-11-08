from gym_virtual_quant_trading.envs.BaseTradingEnv import BaseTradingEnv, BaseTradingEnvConfig

class PaperTradingEnvConfig(BaseTradingEnvConfig):
    TRADED_SYMBOLS = None
    MAX_PORTFOLIO_SIZE = None

class PaperTradingEnv(BaseTradingEnv):
    """Trading environment for backtesting, demo trading

    TODO: Add description
    Args:
        BaseTradingEnv ([type]): [description]
    """

    def __init__(self, paperTradingEnvConfig):
        """Initialize paper trading environment

        Args:
            paperTradingEnvConfig (PaperTradingEnvConfig): configuration
        """
        super(PaperTradingEnv, self).__init__(paperTradingEnvConfig)
        if not isinstance(paperTradingEnvConfig, PaperTradingEnvConfig):
            raise ValueError("paperTradingEnvConfig must be of type PaperTradingEnvConfig")

        self._config = paperTradingEnvConfig # type: PaperTradingEnvConfig

    def step(self, symbol, action, amount):
        """Perform trade operation on the market

        Args:
            symbol (str): Symbol to trade
            action (BaseTradingEnvActions): Type of the trade
            amount (float): Amount to trade

        Returns:
            float: reward for executed action
        """
        return super(PaperTradingEnv, self).step(symbol, action, amount)

    def reset(self):
        """Resets environment before next episode"""
        return super().reset()