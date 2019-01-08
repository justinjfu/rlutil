from libcpp.map cimport map


cdef struct TimeStep:
    int state
    double reward
    bint done


cdef class TabularEnv(object):
    cdef public int num_states
    cdef public int num_actions
    cdef public observation_space
    cdef public action_space
    cdef int _state
    cdef public dict initial_state_distribution
    cdef map[int, double] _transition_map
    cpdef transitions(self, int state, int action)
    cdef map[int, double] transitions_cy(self, int state, int action)
    cpdef double reward(self, int state, int action, int next_state)
    cpdef observation(self, int state)
    cpdef step(self, int action)
    cpdef TimeStep step_state(self, int action)
    cpdef reset(self)
    cpdef int reset_state(self)
    cpdef transition_matrix(self)
    cpdef reward_matrix(self)
    cpdef set_state(self, int state)
    cpdef int get_state(self)
    cpdef render(self)

cdef class CliffwalkEnv(TabularEnv):
    cdef double transition_noise

cdef class RandomTabularEnv(TabularEnv):
    cdef double[:,:,:] _transition_matrix
    cdef double[:,:] _reward_matrix
