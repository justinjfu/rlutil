from rlutil.envs.gridcraft.grid_env import GridEnv, ACT_RIGHT
from rlutil.envs.gridcraft.mazes import *
from rlutil.envs.gridcraft.wrappers import RandomObsWrapper

import unittest


class GridEnvCyTest(unittest.TestCase):
    def testRun(self):
        env = GridEnv(MAZE_LAVA, teps=0.2)
        env = RandomObsWrapper(env, 5)
        obs = env.reset()

        for _ in range(20):
            # env.render()
            obs = env.step(ACT_RIGHT)


if __name__ == "__main__":
    unittest.main()