import itertools

import gym
from gym.spaces import Box
from gym.spaces import Discrete

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from rllab.misc import logger

from rlutil.envs.env_utils import flat_to_one_hot
import rlutil.log_utils as log_utils
from rlutil.math_utils import np_seed

class LQREnv(gym.Env):
    def __init__(self, 
            initial_state, dA, dynamics_matrix, 
            rew_Q, rew_R, rew_q, 
            rew_bias=0,
            frameskip=10,
            ):
        super(LQREnv, self).__init__()
        self.x0 = initial_state
        self.dO = self.x0.shape[0]
        self.dA = dA

        self.dynamics = dynamics_matrix
        self.rew_Q = rew_Q
        self.rew_q = rew_q
        self.rew_R = rew_R
        self.rew_bias = rew_bias
        self.frameskip = frameskip

        self.__observation_space = Box(-1.0, 1.0, shape=(self.dO,))
        self.__action_space = Box(-1.0, 1.0, shape=(self.dA,))
        self.__x = None


    def _wrap_obs(self, x):
        return x

    def eval_reward(self, x, u):
        return x.T.dot(self.rew_Q).dot(x) + self.rew_q.dot(x) + u.T.dot(self.rew_R).dot(u) + self.rew_bias

    def reset(self):
        self.__x = self.x0
        return self._wrap_obs(self.__x)

    def step(self, u):
        r = self.eval_reward(self.__x, u)
        x = self.__x
        for _ in range(self.frameskip):
            x = self.dynamics.dot(np.r_[x, u])
        self.__x = x
        done = False
        return self._wrap_obs(self.__x), r, done, {}

    def log_diagnostics(self, paths):
        pass

    @property
    def action_space(self):
        return self.__action_space

    @property
    def observation_space(self):
        return self.__observation_space


class PointmassEnvVelocity(LQREnv):
    def __init__(self, initial_pos=None, goal_pos=None, action_penalty=1e-2, dt=0.1, sim_steps=1):
        if initial_pos is None:
            initial_pos = np.zeros(2)
        if goal_pos is None:
            goal_pos = np.zeros(2)

        initial_state = initial_pos
        rew_R = - np.eye(2) * action_penalty
        rew_Q = - np.eye(2)
        rew_q = 2*goal_pos
        rew_bias = -np.inner(goal_pos, goal_pos)

        d_sim = dt/sim_steps 
        dynamics_matrix = np.array([
            [1.0, 0.0, d_sim, 0.0],
            [0.0, 1.0, 0.0, d_sim]])
        super(PointmassEnvVelocity, self).__init__(
            initial_state, 2, dynamics_matrix,
            rew_Q, rew_R, rew_q,
            rew_bias=rew_bias,
            frameskip=sim_steps
        )


class PointmassEnvTorque(LQREnv):
    def __init__(self, initial_pos=None, goal_pos=None, 
            action_penalty=1e-2, dt=0.1, sim_steps=1, gains=10):
        if initial_pos is None:
            initial_pos = np.zeros(2)
        if goal_pos is None:
            goal_pos = np.zeros(2)
        initial_state = np.r_[initial_pos, np.zeros(2)]
        rew_R = - np.eye(2) * action_penalty
        rew_Q = - np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            ])
        rew_q = np.r_[2*goal_pos, np.zeros(2)]
        rew_bias = -np.inner(goal_pos, goal_pos)

        d_sim = dt/sim_steps 
        dynamics_matrix = np.array([
            [1.0, 0.0, d_sim, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, d_sim, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, gains*0.5*d_sim**2, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, gains*0.5*d_sim**2],
            ])
        super(PointmassEnvTorque, self).__init__(
            initial_state, 2, dynamics_matrix,
            rew_Q, rew_R, rew_q,
            rew_bias=rew_bias,
            frameskip=sim_steps
        )


class PointmassEnvVision(PointmassEnvTorque):
    def __init__(self, **kwarwgs, im_width=64, im_height=64):
        self.w = im_width
        self.h = im_height
        super(PointmassEnvVision, self).__init__(**kwargs)

    def _wrap_obs(self, x):
        raise NotImplementedError()


if __name__ == '__main__':
    env = PointmassEnvTorque()
    env.reset()
    for _ in range(100):
        print(env.step(env.action_space.sample()))

