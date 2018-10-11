# distutils: language=c++
import sys

import numpy as np
import gym
import gym.spaces

from rlutil.envs.gridcraft.grid_spec_cy cimport TileType, GridSpec
from rlutil.envs.gridcraft.grid_spec_cy import RENDER_DICT
from rlutil.envs.gridcraft.utils import one_hot_to_flat, flat_to_one_hot
from rlutil.envs.tabular_cy cimport tabular_env

from libcpp.map cimport map, pair

cdef int ACT_NOOP = 0
cdef int ACT_UP = 1
cdef int ACT_DOWN = 2
cdef int ACT_LEFT = 3
cdef int ACT_RIGHT = 4

cdef class GridEnv(tabular_env.TabularEnv):
    def __init__(self, 
                 GridSpec gridspec):
        start_xys = np.array(np.where(gridspec.data == TileType.START)).T
        start_idxs = [gridspec.xy_to_idx(pair[int, int](xy[0], xy[1])) for xy in start_xys]
        initial_state_distr = {state: 1.0/len(start_idxs) for state in start_idxs}
        super(GridEnv, self).__init__(len(gridspec), 5, initial_state_distr)
        self.gs = gridspec

    cdef map[int, double] transitions_cy(self, int state, int action):
        cdef int new_x, new_y
        self._transition_map.clear()
        xy = self.gs.idx_to_xy(state)
        tile_type = self.gs.get_value(xy)
        if tile_type == TileType.LAVA: # Lava gets you stuck
            self._transition_map.insert(pair[int, double](state, 1.0))
        else:
            new_x = xy.first
            new_y = xy.second
            if action == ACT_RIGHT:
                new_x += 1
            elif action == ACT_LEFT:
                new_x -= 1
            elif action == ACT_UP:
                new_y -= 1
            elif action == ACT_DOWN:
                new_y += 1
            new_x = min(max(new_x, 0), self.gs.height)
            new_y = min(max(new_y, 0), self.gs.width)
            self._transition_map.insert(pair[int, double](self.gs.xy_to_idx(pair[int, int](new_x, new_y)), 1.0))
        return self._transition_map

    cpdef double reward(self, int state, int action, int next_state):
        cdef TileType tile
        tile = self.gs.get_value(self.gs.idx_to_xy(state))
        if tile == TileType.REWARD:
            return 1.0
        elif tile == TileType.LAVA:
            return -1.0
        return 0.0

    cpdef render(self):
        ostream = sys.stdout
        state = self.get_state()
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