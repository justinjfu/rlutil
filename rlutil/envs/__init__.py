from gym.envs.registration import register
import logging

from rlutil.envs.gridcraft.mazes import *
from rlutil.envs.tabular.simple_env import random_env_register

LOGGER = logging.getLogger(__name__)

_REGISTERED = False
def register_envs():
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    LOGGER.info("Registering custom gym environments")
    register(id='GridMaze1-v0', entry_point='baselines.envs.gridcraft.grid_env:GridEnv',
             kwargs={'one_hot': True, 'add_eyes': True, 'gridspec': MAZE1})
    register(id='Tabular32x4-v0', entry_point='baselines.envs.tabular.simple_env:DiscreteEnv',
             kwargs=random_env_register(32, 4, seed=0))
    register(id='Tabular8x2-v0', entry_point='baselines.envs.tabular.simple_env:DiscreteEnv',
             kwargs=random_env_register(8, 2, seed=0))
    register(id='TabularDeterm32x8-v0', entry_point='baselines.envs.tabular.simple_env:DiscreteEnv',
             kwargs=random_env_register(32, 8, seed=0, deterministic=True))
    register(id='TabularDeterm8x4-v0', entry_point='baselines.envs.tabular.simple_env:DiscreteEnv',
             kwargs=random_env_register(8, 4, seed=0, deterministic=True))
    register(id='TabularDeterm64x8-v0', entry_point='baselines.envs.tabular.simple_env:DiscreteEnv',
             kwargs=random_env_register(64, 8, seed=0, deterministic=True))
    register(id='TabularDetermObs32x8-v0', entry_point='baselines.envs.tabular.simple_env:DiscreteEnv',
             kwargs=random_env_register(32, 8, seed=0, deterministic=True, dim_obs=4))
    register(id='TabularDetermObs8x4-v0', entry_point='baselines.envs.tabular.simple_env:DiscreteEnv',
             kwargs=random_env_register(8, 4, seed=0, deterministic=True, dim_obs=4))
    register(id='TabularDetermObs64x8-v0', entry_point='baselines.envs.tabular.simple_env:DiscreteEnv',
             kwargs=random_env_register(64, 8, seed=0, deterministic=True, dim_obs=4))


    LOGGER.info("Finished registering custom gym environments")
