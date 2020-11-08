from gym_virtual_quant_trading.envs.PaperTradingEnv import PaperTradingEnv, PaperTradingEnvConfig

env_cfg = PaperTradingEnvConfig()
env = PaperTradingEnv(env_cfg)
print(next(env._config.DATA_SOURCE))