import unittest

from rlutil.envs.gridcraft import grid_env_cy
from rlutil.envs.gridcraft import grid_spec_cy
from rlutil.envs.gridcraft import grid_env
from rlutil.envs.gridcraft import grid_spec
from rlutil.envs.gridcraft import mazes

class GridEnvCyTest(unittest.TestCase):
    def setUp(self):
        maze = "SOO\\OOR"
        gs_cy = grid_spec_cy.spec_from_string(maze)
        gs_py = grid_spec.spec_from_string(maze)
        self.cy_env = grid_env_cy.GridEnv(gs_cy)
        self.py_env = grid_env.GridEnv(gs_py)
    
    def testReset(self):
        cy_s = self.cy_env.reset()
        py_s = self.py_env.reset()
        self.assertEqual(cy_s, py_s)
    
    def test_transitions(self):
        cy_s = self.cy_env.reset()
        py_s = self.py_env.reset()
        for _ in range(10):
            action = self.py_env.action_space.sample()
            cy_s, _, _, _ = self.cy_env.step(action)
            py_s, _, _, _ = self.py_env.step(action)
            self.assertEqual(cy_s, py_s)

if __name__ == "__main__":
    unittest.main()