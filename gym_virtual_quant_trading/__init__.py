from gym.envs.registration import register

register(
    id='virtual-quant-paper-trading-v0',
    entry_point='gym_trading.envs:PaperTradingEnv',
)