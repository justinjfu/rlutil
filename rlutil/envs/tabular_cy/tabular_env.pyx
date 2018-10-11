# distutils: language=c++
"""Base class for cython-based tabular envs.

Subclasses should implement the transitions_cy,  reward methods.
An example environment is provided in LinkedListEnv
"""
import gym
import gym.spaces
import numpy as np
import cython
from rlutil.envs.tabular_cy.tabular_env cimport TimeStep

from libc.math cimport fmin
from libcpp.map cimport map, pair
from libc.stdlib cimport rand
cdef extern from "limits.h":
    int INT_MAX
from cython.operator cimport dereference, preincrement


@cython.cdivision(True)
cdef inline int sample_int(map[int, double] transitions):
    cdef float randnum = rand() / float(INT_MAX)
    cdef float total = 0
    transitions_end = transitions.end()
    transitions_it = transitions.begin()
    while transitions_it != transitions_end:
        ns = dereference(transitions_it).first
        p = dereference(transitions_it).second
        
        if (p+total) >= randnum:
            return ns
        total += p
        preincrement(transitions_it)


cdef class TabularEnv(object):
    """Base class for tabular environments.

    States and actions are represented as integers ranging from
    [0,  self.num_states) or [0, self.num_actions), respectively.

    Args:
      num_states: Size of the state space.
      num_actions: Size of the action space.
      initial_state_distribution: A dictionary from states to
        probabilities representing the initial state distribution.
    """

    def __init__(self,
                 int num_states,
                 int num_actions,
                 dict initial_state_distribution):
        self._state = -1
        self.observation_space = gym.spaces.Discrete(num_states)
        self.action_space = gym.spaces.Discrete(num_actions)
        self.num_states = num_states
        self.num_actions = num_actions
        self.initial_state_distribution = initial_state_distribution

    cpdef transitions(self, int state, int action):
        """Computes transition probabilities p(ns|s,a).

        Args:
          state:
          action:

        Returns:
          A python dict from {next state: probability}.
          (Omitted states have probability 0)
        """
        return dict(self.transitions_cy(state, action))

    cdef map[int, double] transitions_cy(self, int state, int action):
        self._transition_map.clear()
        self._transition_map.insert(pair[int, double](state, 1.0))
        return self._transition_map

    cpdef double reward(self, int state, int action, int next_state):
        """Return the reward

        Args:
          state:
          action: 
          next_state: 
        """
        return 0.0

    cpdef observation(self, int state):
        """Computes observation for a given state.

        Args:
          state: 

        Returns:
          observation: Agent's observation of state, conforming with observation_space
        """
        return state

    cpdef step(self, int action):
        """Simulates the environment by one timestep.

        Args:
          action: Action to take

        Returns:
          observation: Next observation
          reward: Reward incurred by agent
          done: A boolean indicating the end of an episode
          info: A debug info dictionary.
        """
        infos = {'state': self.get_state()}
        next_state, reward, done = self.step_state(action)
        nobs = self.observation(next_state)
        return nobs, reward, done, infos

    @cython.infer_types(True)
    cdef TimeStep step_state(self, int action):
        """Simulates the environment by one timestep, returning the state id
        instead of the observation.

        Args:
          action: Action taken by the agent.

        Returns:
          state: Next state
          reward: Reward incurred by agent
          done: A boolean indicating the end of an episode
          info: A debug info dictionary.
        """
        cdef int next_state
        transitions = self.transitions_cy(self._state, action)
        #next_state = np.random.choice(
        #    list(transitions.keys()), p=list(transitions.values()))
        next_state = sample_int(transitions)
        reward = self.reward(self.get_state(), action, next_state)
        self._state = next_state
        return TimeStep(next_state, reward, False)

    cpdef reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns:
          observation (object): The agent's initial observation.
        """
        initial_state = self.reset_state()
        return self.observation(initial_state)

    cdef int reset_state(self):
        """Resets the state of the environment and returns an initial state.

        Returns:
          state: The agent's initial state
        """
        initial_states = list(self.initial_state_distribution.keys())
        initial_probs = list(self.initial_state_distribution.values())
        initial_state = np.random.choice(initial_states, p=initial_probs)
        self._state = initial_state
        return initial_state

    @cython.boundscheck(False)
    cpdef transition_matrix(self):
        """Constructs this environment's transition matrix.

        Returns:
          A dS x dA x dS array where the entry transition_matrix[s, a, ns]
          corrsponds to the probability of transitioning into state ns after taking
          action a from state s.
        """
        cdef int next_s
        ds = self.num_states
        da = self.num_actions
        transition_matrix_np = np.zeros((ds, da, ds))
        cdef double[:, :, :] transition_matrix = transition_matrix_np
        for s in range(ds):
            for a in range(da):
                transitions = self.transitions_cy(s, a)
                transitions_end = transitions.end()
                transitions_it = transitions.begin()
                while transitions_it != transitions_end:
                    next_s = dereference(transitions_it).first
                    prob = dereference(transitions_it).second
                    transition_matrix[s, a, next_s] = prob
                    preincrement(transitions_it)
        return transition_matrix_np

    @cython.boundscheck(False)
    cpdef reward_matrix(self):
        """Constructs this environment's reward matrix.

        Returns:
          A dS x dA x dS numpy array where the entry reward_matrix[s, a, ns]
          reward given to an agent when transitioning into state ns after taking
          action s from state s.
        """
        ds = self.num_states
        da = self.num_actions
        rew_matrix_np = np.zeros((ds, da, ds))
        cdef double[:, :, :] rew_matrix = rew_matrix_np
        for s in range(ds):
            for a in range(da):
                for ns in range(ds):
                    rew_matrix[s, a, ns] = self.reward(s, a, ns)
        return rew_matrix_np

    cpdef set_state(self, int state):
        """Set the agent's internal state."""
        self._state = state

    cpdef int get_state(self):
        """Return the agent's internal state."""
        return self._state
    
    cpdef render(self):
        """Render the current state of the environment."""
        pass


cdef class LinkedListEnv(TabularEnv):
    """An example env where an agent can move along a sequence of states. There is
    a chance that the agent may jump back to the initial state.

    Action 0 moves the agent back to start, and action 1 to the next state.
    The agent only receives reward in the final state.

    Args:
      num_states: Number of states 
      transition_noise: A float in [0, 1] representing the chance that the
        agent will be transported to the start state.
    """

    def __init__(self, int num_states=3, double transition_noise=0.0):
        super(LinkedListEnv, self).__init__(num_states, 2, {0: 1.0})
        self.transition_noise = transition_noise

    cdef map[int, double] transitions_cy(self, int state, int action):
        self._transition_map.clear()
        if action == 0:
            self._transition_map.insert(pair[int, double](0, 1.0))
        else:
            self._transition_map.insert(
                pair[int, double](0, self.transition_noise))
            self._transition_map.insert(pair[int, double](
                    int(fmin(state + 1, self.num_states - 1)), 1.0 - self.transition_noise))
        return self._transition_map

    cpdef double reward(self, int state, int action, int next_state):
        if state == self.num_states - 1:
            return 1.0
        else:
            return 0.0
