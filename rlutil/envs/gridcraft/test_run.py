from rlutil.envs.gridcraft.grid_env import GridEnv, ACT_RIGHT
from rlutil.envs.gridcraft.mazes import *
from rlutil.envs.gridcraft.wrappers import RandomObsWrapper

env = GridEnv(MAZE_LAVA, teps=0.2)
env = RandomObsWrapper(env, 5)
obs = env.reset()
print(obs)

for _ in range(20):
    env.render()
    #env.step(env.action_space.sample())
    obs = env.step(ACT_RIGHT)
    print(obs)
