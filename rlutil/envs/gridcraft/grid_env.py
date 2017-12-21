import inspect
import itertools
import sys

import numpy as np
import gym
import gym.spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from rllab.misc import logger

import rlutil.log_utils as log_utils
from rlutil.envs.gridcraft.grid_spec import *
from rlutil.envs.gridcraft.utils import one_hot_to_flat, flat_to_one_hot

ACT_NOOP = 0
ACT_UP = 1
ACT_DOWN = 2
ACT_LEFT = 3
ACT_RIGHT = 4
ACT_DICT = {
    ACT_NOOP: [0,0],
    ACT_UP: [0, -1],
    ACT_LEFT: [-1, 0],
    ACT_RIGHT: [+1, 0],
    ACT_DOWN: [0, +1]
}
ACT_TO_STR = {
    ACT_NOOP: 'NOOP',
    ACT_UP: 'UP',
    ACT_LEFT: 'LEFT',
    ACT_RIGHT: 'RIGHT',
    ACT_DOWN: 'DOWN'
}

class TransitionModel(object):
    def __init__(self, gridspec, eps=0.2):
        self.gs = gridspec
        self.eps = eps

    def get_aprobs(self, s, a):
        # TODO: could probably output a matrix over all states...
        legal_moves = self.__get_legal_moves(s)
        p = np.zeros(len(ACT_DICT))
        p[legal_moves] = self.eps / (len(legal_moves))
        if a in legal_moves:
            p[a] += 1.0-self.eps
        else:
            #p = np.array([1.0,0,0,0,0])  # NOOP
            p[ACT_NOOP] += 1.0-self.eps
        return p

    def __get_legal_moves(self, s):
        xy = np.array(self.gs.idx_to_xy(s))
        moves = [move for move in ACT_DICT if not self.gs.out_of_bounds(xy+ACT_DICT[move])
                                             and self.gs[xy+ACT_DICT[move]] != WALL]

        #print 'xy:', s, xy
        #print [xy+ACT_DICT[move] for move in ACT_DICT]
        #print 'legal:', [ACT_TO_STR[move] for move in moves]

        return moves


class RewardFunction(object):
    def __init__(self, rew_map=None):
        if rew_map is None:
            rew_map = {
                REWARD: 1.0,
                REWARD2: 2.0,
                REWARD3: 4.0,
                REWARD4: 8.0,
                LAVA: -1.0,
            }
        self.rew_map = rew_map

    def __call__(self, gridspec, s, a, ns):
        val = gridspec[gridspec.idx_to_xy(s)]
        if val in self.rew_map:
            return self.rew_map[val]
        return 0.0


class GridEnv(gym.Env):
    def __init__(self, gridspec, 
                 tiles=TILES,
                 rew_fn=None,
                 teps=0.0, 
                 max_timesteps=None):
        self._env_args = {'teps': teps, 'max_timesteps': max_timesteps}
        self.gs = gridspec
        self.model = TransitionModel(gridspec, eps=teps)
        if rew_fn is None:
            rew_fn = RewardFunction()
        self.rew_fn = rew_fn
        self.possible_tiles = tiles
        self.max_timesteps = max_timesteps
        self._timestep = 0
        self._true_q = None  # q_vals for debugging
        super(GridEnv, self).__init__()

    def get_transitions(self, s, a):
        tile_type = self.gs[self.gs.idx_to_xy(s)]
        if tile_type == LAVA: # Lava gets you stuck
            return {s: 1.0}

        aprobs = self.model.get_aprobs(s, a)
        t_dict = {}
        for sa in range(5):
            if aprobs[sa] > 0:
                next_s = self.gs.idx_to_xy(s) + ACT_DICT[sa]
                next_s_idx = self.gs.xy_to_idx(next_s)
                t_dict[next_s_idx] = t_dict.get(next_s_idx, 0.0) + aprobs[sa]
        return t_dict


    def step_stateless(self, s, a, verbose=False):
        aprobs = self.model.get_aprobs(s, a)
        samp_a = np.random.choice(range(5), p=aprobs)

        next_s = self.gs.idx_to_xy(s) + ACT_DICT[samp_a]
        tile_type = self.gs[self.gs.idx_to_xy(s)]
        if tile_type == LAVA: # Lava gets you stuck
            next_s = self.gs.idx_to_xy(s)

        next_s_idx = self.gs.xy_to_idx(next_s)
        rew = self.rew_fn(self.gs, s, samp_a, next_s_idx)

        if verbose:
            print('Act: %s. Act Executed: %s' % (ACT_TO_STR[a], ACT_TO_STR[samp_a]))
        return next_s_idx, rew

    def step(self, a, verbose=False):
        ns, r = self.step_stateless(self.__state, a, verbose=verbose)
        traj_infos = {}
        self.__state = ns
        obs = flat_to_one_hot(ns, len(self.gs))

        done = False
        self._timestep += 1
        if self.max_timesteps is not None:
            if self._timestep >= self.max_timesteps:
                done = True
        return obs, r, done, traj_infos

    def reset(self):
        start_idxs = np.array(np.where(self.gs.spec == START)).T
        start_idx = start_idxs[np.random.randint(0, start_idxs.shape[0])]
        start_idx = self.gs.xy_to_idx(start_idx)
        self.__state =start_idx
        self._timestep = 0
        return flat_to_one_hot(start_idx, len(self.gs))

    def get_tile(self, obs):
        idx = self.obs_to_state(obs)
        return self.gs.get_value(idx)

    def render(self, close=False, ostream=sys.stdout):
        if close:
            return

        state = self.__state
        ostream.write('-'*(self.gs.width+2)+'\n')
        for h in range(self.gs.height):
            ostream.write('|')
            for w in range(self.gs.width):
                if self.gs.xy_to_idx((w,h)) == state:
                    ostream.write('*')
                else:
                    val = self.gs[w, h]
                    ostream.write(RENDER_DICT[val])
            ostream.write('|\n')
        ostream.write('-' * (self.gs.width + 2)+'\n')

    @property
    def action_space(self):
        return gym.spaces.Discrete(5)

    @property
    def observation_space(self):
        dO = len(self.gs)
        return gym.spaces.Box(0,1,shape=dO)

    def log_diagnostics(self, paths):
        Ntraj = len(paths)
        acts = np.array([traj['actions'] for traj in paths])
        obs = np.array([traj['observations'] for traj in paths])

        state_count = np.sum(obs, axis=1)
        states_visited = np.sum(state_count>0, axis=-1)
        #log states visited
        logger.record_tabular('AvgStatesVisited', np.mean(states_visited))

    def plot_trajs(self, paths, dirname=None, itr=0):
        plt.figure()
        # draw walls
        ax = plt.gca()
        wall_positions = self.gs.find(WALL)
        for i in range(wall_positions.shape[0]):
            wall_xy = wall_positions[i,:]
            wall_xy[1] = self.gs.height-wall_xy[1]-1
            ax.add_patch(Rectangle(wall_xy-0.5, 1, 1))
        #plt.scatter(wall_positions[:,0], wall_positions[:,1], color='k')

        val_to_color = {
            REWARD: (0,0.2,0.0),
            REWARD2: (0.0, 0.5, 0.0),
            REWARD3: (0.0, 1.0, 0.0),
            REWARD4: (1.0, 0.0, 1.0),
            START: 'b',
        }
        for key in val_to_color:
            rew_positions = self.gs.find(key)
            plt.scatter(rew_positions[:,0], self.gs.height-rew_positions[:,1]-1, color=val_to_color[key])

        for path in paths:
            obses = self.obs_to_state(path['observations'])
            xys = self.gs.idx_to_xy(obses)
            # plot x, y positions
            plt.plot(xys[:,0], self.gs.height-xys[:,1]-1)

        ax.set_xticks(np.arange(-1, self.gs.width+1, 1))
        ax.set_yticks(np.arange(-1, self.gs.height+1, 1))
        plt.grid()

        if dirname is not None:
            log_utils.record_fig('trajs_itr%d'%itr, subdir=dirname, rllabdir=True)
        else:
            plt.show()

    def debug_qval(self, q_func, obses=None, acts=None, gamma=0.95):
        # q_func: f(s, a) => double

        # get all states
        _obses = []
        _acts = []
        for (x, y, a) in itertools.product(range(self.gs.width), range(self.gs.height), range(5)):
            obs = flat_to_one_hot(self.gs.xy_to_idx((x, y)), ndim=len(self.gs))
            act = a #flat_to_one_hot(a, ndim=5)
            _obses.append(obs)
            _acts.append(act)
        _obses = np.array(_obses)
        _acts = np.array(_acts)

        # eval q
        qvals = q_func(_obses, _acts)

        # true q
        if self._true_q is None:
            from rlutil.envs.gridcraft.true_qvalues import load_qvals
            self._true_q = load_qvals(self.gs, self._env_args, gamma=gamma)
        true_qvals = self._true_q._q_vec

        # log errors
        q_err = np.abs(qvals - true_qvals)
        logger.record_tabular('QStarErrorMean', np.mean(q_err))
        logger.record_tabular('QStarErrorMax', np.max(q_err))
        logger.record_tabular('QStarErrorMin', np.min(q_err))

        if obses is not None and acts is not None:
            qvals = q_func(obses, acts)
            true_qvals = self._true_q(obses, acts)
            q_err = np.abs(qvals - true_qvals)
            logger.record_tabular('QStarSampledErrorMean', np.mean(q_err))
            logger.record_tabular('QStarSampeldErrorMax', np.max(q_err))
            logger.record_tabular('QStarSampledErrorMin', np.min(q_err))
        else:
            logger.record_tabular('QStarSampledErrorMean', float('NaN'))
            logger.record_tabular('QStarSampeldErrorMax', float('NaN'))
            logger.record_tabular('QStarSampledErrorMin', float('NaN'))


    def plot_costs(self, paths, cost_fn, dirname=None, itr=0, policy=None,
                   use_traj_paths=False):
        #costs = cost_fn.eval(paths)
        if self.gs.width*self.gs.height > 600:
            use_text = False
        else:
            use_text = True


        if not use_traj_paths:
            # iterate through states, and each action - makes sense for non-rnn costs
            import itertools
            obses = []
            acts = []
            for (x, y, a) in itertools.product(range(self.gs.width), range(self.gs.height), range(5)):
                obs = self.state_to_obs(self.gs.xy_to_idx((x, y)))
                act = flat_to_one_hot(a, ndim=5)
                obses.append(obs)
                acts.append(act)
            path = {'observations': np.array(obses), 'actions': np.array(acts)}
            if policy is not None:
                if hasattr(policy, 'set_env_infos'):
                    policy.set_env_infos(path.get('env_infos', {}))
                actions, agent_infos = policy.get_actions(path['observations'])
                path['agent_infos'] = agent_infos
            paths = [path]

        plots = cost_fn.debug_eval(paths, policy=policy)
        for plot in plots:
            plots[plot] = plots[plot].squeeze()

        for plot in plots:
            data = plots[plot]

            plotter = TabularQValuePlotter(self.gs.width, self.gs.height, text_values=use_text)
            for i, (x, y, a) in enumerate(itertools.product(range(self.gs.width), range(self.gs.height), range(5))):
                plotter.set_value(x, self.gs.height-y-1, a, data[i])
            plotter.make_plot()
            if dirname is not None:
                log_utils.record_fig('%s_itr%d'%(plot, itr), subdir=dirname, rllabdir=True)
            else:
                plt.show()

