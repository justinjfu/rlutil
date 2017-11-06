import numpy as np
from rlutil.envs.gridcraft import REWARD
from rlutil.envs.wrappers import ObsWrapper
from gym.spaces import Box


class EyesWrapper(ObsWrapper):
    def __init__(self, env, range=4, types=(REWARD,), angle_thresh=0.8):
        super(EyesWrapper, self).__init__(env)
        self.types = types
        self.range = range
        self.angle_thresh = angle_thresh

        eyes_low = np.ones(5*len(types))
        eyes_high = np.ones(5*len(types))
        low = np.r_[env.observation_space.low, eyes_low]
        high = np.r_[env.observation_space.high, eyes_high]
        self.__observation_space = Box(low, high)

    def wrap_obs(self, obs, info=None):
        gs = self.env.gs  # grid spec
        xy = gs.idx_to_xy(self.env.obs_to_state(obs))
        #xy = np.array([x, y])

        extra_obs = []
        for tile_type in self.types:
            idxs = gs.find(tile_type).astype(np.float32)  # N x 2
            # gather all idxs that are close
            diffs = idxs-np.expand_dims(xy, axis=0)
            dists = np.linalg.norm(diffs, axis=1)
            valid_idxs = np.where(dists <= self.range)[0]
            if len(valid_idxs) == 0:
                eye_data = np.array([0,0,0,0,0], dtype=np.float32)
            else:
                diffs = diffs[valid_idxs, :]
                dists = dists[valid_idxs]+1e-6
                cosines = diffs[:,0]/dists
                cosines = np.r_[cosines, 0]
                sines = diffs[:,1]/dists
                sines = np.r_[sines, 0]
                on_target = 0.0
                if np.any(dists<=1.0):
                    on_target = 1.0
                eye_data = np.abs(np.array([on_target, np.max(cosines), np.min(cosines), np.max(sines), np.min(sines)]))
                eye_data[np.where(eye_data<=self.angle_thresh)] = 0
            extra_obs.append(eye_data)
        extra_obs = np.concatenate(extra_obs)
        obs = np.r_[obs, extra_obs]
        #if np.any(np.isnan(obs)):
        #    import pdb; pdb.set_trace()
        return obs

    def unwrap_obs(self, obs, info=None):
        if len(obs.shape) == 1:
            return obs[:-5*len(self.types)]
        else:
            return obs[:,:-5*len(self.types)]

    @property
    def observation_space(self):
        return self.__observation_space
