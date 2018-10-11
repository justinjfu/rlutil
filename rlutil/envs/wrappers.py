import numpy as np
from gym import Env, logger
from gym import error
from gym.spaces import Box


class Wrapper(Env):
    # Clear metadata so by default we don't override any keys.
    metadata = {}

    _owns_render = False

    # Make sure self.env is always defined, even if things break
    # early.
    env = None
    _COUNTER = 0

    def __init__(self, env=None):
        Wrapper._COUNTER += 1
        self._wrapper_id = Wrapper._COUNTER
        #Serializable.quick_init(self, locals())
        self.env = env
        # Merge with the base metadata
        metadata = self.metadata
        self.metadata = self.env.metadata.copy()
        self.metadata.update(metadata)

        self.__action_space = self.env.action_space
        self.__observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self._spec = None #self.env.spec
        self._unwrapped = self.env.unwrapped
        self.wrapped_observation_space = env.observation_space

        self._update_wrapper_stack()
        if env and hasattr(env, '_configured') and env._configured:
            logger.warning("Attempted to wrap env %s after .configure() was called.", env)

    def _update_wrapper_stack(self):
        """
        Keep a list of all the wrappers that have been appended to the stack.
        """
        self._wrapper_stack = getattr(self.env, '_wrapper_stack', [])
        self._check_for_duplicate_wrappers()
        self._wrapper_stack.append(self)

    def _check_for_duplicate_wrappers(self):
        """Raise an error if there are duplicate wrappers. Can be overwritten by subclasses"""
        if self.class_name() in [wrapper.class_name() for wrapper in self._wrapper_stack]:
            raise error.DoubleWrapperError("Attempted to double wrap with Wrapper: {}".format(self.class_name()))

    @property
    def wrapped_env(self):
        if isinstance(self.env, Wrapper):
            return self.env.wrapped_env
        else:
            return self.env

    @property
    def action_space(self):
        return self.__action_space

    @property
    def observation_space(self):
        return self.__observation_space

    @classmethod
    def class_name(cls):
        return cls.__name__

    @property
    def is_latent_env_wrapper(self):
        return False

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self, mode='human', close=False):
        if self.env is None:
            return
        return self.env.render(mode, close)

    def _close(self):
        if self.env is None:
            return
        return self.env.close()

    def _configure(self, *args, **kwargs):
        return self.env.configure(*args, **kwargs)

    def _seed(self, seed=None):
        return self.env.seed(seed)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)

    @property
    def spec(self):
        if self._spec is None:
            self._spec = self.env.spec
        return self._spec

    @spec.setter
    def spec(self, spec):
        # Won't have an env attr while in the __new__ from gym.Env
        if self.env is not None:
            self.env.spec = spec
        self._spec = spec

    def log_diagnostics(self, paths):
        if hasattr(self.env, 'log_diagnostics'):
            self.env.log_diagnostics(paths)


class ObsWrapper(Wrapper):
    def __init__(self, env):
        super(ObsWrapper, self).__init__(env)

    def wrap_obs(self, obs, info=None):
        raise NotImplementedError()

    def unwrap_obs(self, obs, info=None):
        raise NotImplementedError()

    def _step(self, action):
        obs, r, done, infos = self.env.step(action)
        return self.wrap_obs(obs, info=infos), r, done, infos

    def _reset(self, env_info=None):
        if env_info is None:
            env_info = {}
        obs = self.env.reset()
        return self.wrap_obs(obs, info=env_info)

    def wrap_obs_multi(self, obses, info=None):
        if len(obses.shape) == 2:
            return np.array([self.wrap_obs(obses[i], info=info) for i in range(obses.shape[0])])
        else:
            return self.wrap_obs(obses, info=info)

    def unwrap_paths(self, paths):
        new_paths = []
        for path in paths:
            env_infos = path.get('env_infos', {})
            new_paths.append({
                'observations': self.unwrap_obs(path['observations'], info=env_infos),
                'actions': path['actions'],
                'env_infos': env_infos,
                'agent_infos': path.get('agent_infos', {})
            })
        return new_paths

    def wrap_paths(self, paths):
        new_paths = []
        for path in paths:
            env_infos = path.get('env_infos', {})
            new_paths.append({
                'observations': self.wrap_obs_multi(path['observations'], info=env_infos),
                'actions': path['actions'],
                'env_infos': env_infos,
                'agent_infos': path.get('agent_infos', {})
            })
        return new_paths

    def log_diagnostics(self, paths):
        if hasattr(self.env, 'log_diagnostics'):
            self.env.log_diagnostics(self.unwrap_paths(paths))

    def plot_trajs(self, paths, **kwargs):
        if hasattr(self.env, 'plot_trajs'):
            self.env.plot_trajs(self.unwrap_paths(paths), **kwargs)

    def plot_costs(self, paths, cost_fn, policy=None, **kwargs):
        env = self
        if policy is not None:
            class wrap_policy(object):
                def set_env_infos(self, env_info):
                    self.env_info = env_info

                def get_actions(self, observations):
                    env_info = self.env_info
                    delattr(self, 'env_info')
                    return policy.get_actions(env.wrap_obs_multi(observations, info=env_info))
            wrapped_policy = wrap_policy()
        else:
            wrapped_policy = None

        class wrap_cost(object):
            def eval(self, paths, **kwargs):
                return cost_fn.eval(env.wrap_paths(paths), **kwargs)

            def debug_eval(self, paths, **kwargs):
                return cost_fn.debug_eval(env.wrap_paths(paths), **kwargs)
        wrapped_cost = wrap_cost()

        if hasattr(self.env, 'plot_trajs'):
            self.env.plot_costs(self.unwrap_paths(paths), cost_fn=wrapped_cost, policy=wrapped_policy, **kwargs)


class FixedEncodeWrapper(ObsWrapper):
    def __init__(self, env, fixed_encoding):
        #Serializable.quick_init(self, locals())
        super(FixedEncodeWrapper, self).__init__(env)
        self.fixed_encoding = fixed_encoding
        assert isinstance(env.observation_space, Box)
        assert len(env.observation_space.shape) == 1
        assert len(fixed_encoding.shape) == 1

        self.inner_dim = env.observation_space.shape[0]

        low = np.r_[env.observation_space.low, fixed_encoding]
        high = np.r_[env.observation_space.high, fixed_encoding]
        self.__observation_space = Box(low, high)

    def wrap_obs(self, obs, info=None):
        obs = np.r_[obs, self.fixed_encoding]
        return obs

    def unwrap_obs(self, obs, info=None):
        if len(obs.shape) == 1:
            return obs[:self.inner_dim]
        else:
            return obs[:,:self.inner_dim]

    @property
    def observation_space(self):
        return self.__observation_space


class ZeroObsWrapper(ObsWrapper):
    def __init__(self, env, lo, hi):
        super(ZeroObsWrapper, self).__init__(env)
        self.obs_key = 'og_obs'
        self.idxs = slice(lo, hi)

    def wrap_obs(self, obs, info=None):
        info[self.obs_key] = np.array(obs)
        obs[self.idxs] = 0
        return obs

    def unwrap_obs(self, obs, info=None):
        return info[self.obs_key]
