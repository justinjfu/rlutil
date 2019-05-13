import numpy as np
import os.path as osp

from gym import spaces
from gym import Env
from gym.envs import mujoco as gym_mujoco
import mujoco_py
import tempfile
import os
import mako.template
import mako.lookup

from rlutil.logging import logger

MODEL_DIR = osp.join(osp.dirname(gym_mujoco.__file__), 'assets')
BIG = 1e6
DEFAULT_SIZE = 500

def q_inv(a):
    return [a[0], -a[1], -a[2], -a[3]]

def q_mult(a, b): # multiply two quaternion
    w = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]
    i = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2]
    j = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1]
    k = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]
    return [w, i, j, k]

class MujocoEnv(Env):
    FILE = None

    def __init__(self,  file_path=None, template_args=None, frame_skip=1):
        # compile template
        if file_path is None:
            if self.__class__.FILE is None:
                raise "Mujoco file not specified"
            file_path = osp.join(MODEL_DIR, self.__class__.FILE)
        if file_path.endswith(".mako"):
            lookup = mako.lookup.TemplateLookup(directories=[MODEL_DIR])
            with open(file_path) as template_file:
                template = mako.template.Template(
                    template_file.read(), lookup=lookup)
            content = template.render(
                opts=template_args if template_args is not None else {},
            )
            tmp_f, file_path = tempfile.mkstemp(text=True)
            with open(file_path, 'w') as f:
                f.write(content)
            self.model = mujoco_py.load_model_from_path(file_path)
            os.close(tmp_f)
        else:
            self.model = mujoco_py.load_model_from_path(file_path)

        self.sim = mujoco_py.MjSim(self.model)
        self.data = self._data
        self.init_qpos = self._data.qpos
        self.init_qvel = self._data.qvel
        self.init_qacc = self._data.qacc
        self.init_ctrl = self._data.ctrl
        self.qpos_dim = self.init_qpos.size
        self.qvel_dim = self.init_qvel.size
        self.ctrl_dim = self.init_ctrl.size
        self.frame_skip = frame_skip
        self.reset()

        # openai gym
        self.viewer = None
        self._viewers = {}
        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        super(MujocoEnv, self).__init__()

    @property
    def _data(self):
        return self.sim.data

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    @property
    def action_space(self):
        bounds = self.model.actuator_ctrlrange
        lb = bounds[:, 0]
        ub = bounds[:, 1]
        return spaces.Box(lb, ub, dtype=np.float32)

    @property
    def observation_space(self):
        shp = self._get_obs().shape
        ub = BIG * np.ones(shp)
        return spaces.Box(ub * -1, ub, dtype=np.float32)

    @property
    def action_bounds(self):
        return self.action_space.bounds

    def set_state(self, qpos=None, qvel=None):
        if qpos is None:
            qpos = self._data.qpos
        if qvel is None:
            qvel = self._data.qvel
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def reset_model(self):
        qpos = self.init_qpos + \
                               np.random.normal(size=self.init_qpos.shape) * 0.01
        qvel = self.init_qvel + \
                               np.random.normal(size=self.init_qvel.shape) * 0.1
        self.set_state(qpos, qvel)

    def reset(self):
        self.sim.reset()
        self.reset_model()
        return self._get_obs()

    def _get_obs(self):
        return self._get_full_obs()

    def _get_full_obs(self):
        data = self._data
        cdists = np.copy(self.model.geom_margin).flat
        for c in self._data.contact:
            cdists[c.geom2] = min(cdists[c.geom2], c.dist)
        obs = np.concatenate([
            data.qpos.flat,
            data.qvel.flat,
            # data.cdof.flat,
            data.cinert.flat,
            data.cvel.flat,
            # data.cacc.flat,
            data.qfrc_actuator.flat,
            data.cfrc_ext.flat,
            data.qfrc_constraint.flat,
            cdists,
            # data.qfrc_bias.flat,
            # data.qfrc_passive.flat,
            self.dcom.flat,
        ])
        return obs

    @property
    def _state(self):
        return np.concatenate([
            self._data.qpos.flat,
            self._data.qvel.flat
        ])

    @property
    def _full_state(self):
        return np.concatenate([
            self._data.qpos,
            self._data.qvel,
            self._data.qacc,
            self._data.ctrl,
        ]).ravel()

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array' or mode == 'depth_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)
                
            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        if mode == 'rgb_array':
            camera_id = None 
            camera_name = 'track'
            if self.rgb_rendering_tracking and camera_name in self.model.camera_names:
                camera_id = self.model.camera_name2id(camera_name)
            self._get_viewer(mode).render(width, height, camera_id=camera_id)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'depth_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    """
    def release(self):
        # temporarily alleviate the issue (but still some leak)
        from rllab.mujoco_py.mjlib import mjlib
        mjlib.mj_deleteModel(self.model._wrapped)
        mjlib.mj_deleteData(self.data._wrapped)
    """

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self._data.xmat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        #idx = self.model.body_names.index(body_name)
        #return self._data.com_subtree[idx]
        return self._data.get_body_xpos(body_name)

    def get_body_comvel(self, body_name):
        #idx = self.model.body_names.index(body_name)
        #return self.model.body_comvels[idx]
        return self._data.get_body_xvel(body_name)

    def print_stats(self):
        super(MujocoEnv, self).print_stats()
        print("qpos dim:\t%d" % len(self._data.qpos))

    def action_from_key(self, key):
        raise NotImplementedError
