import unittest
import parameterized
import numpy as np

from rlutil.envs.tabular_cy import q_iteration, tabular_env
from rlutil.envs.tabular_cy import q_iteration_py


class QIterationTest(unittest.TestCase):
  def setUp(self):
    self.env = tabular_env.CliffwalkEnv(num_states=3, transition_noise=0.01)

  def test_qiteration(self):
    params = {
        'num_itrs': 50,
        'ent_wt': 1.0,
        'discount': 0.99,
    }
    qvals_py = q_iteration_py.q_iteration_sparse(self.env, **params)
    qvals_cy = q_iteration.q_iteration_sparse(self.env, **params)
    self.assertTrue(np.allclose(qvals_cy, qvals_py))


if __name__ == '__main__':
  unittest.main()
