from libcpp.map cimport map

cdef class TabularEnv(object):
    cdef public int num_states
    cdef public int num_actions
    cdef public observation_space
    cdef public action_space
    cdef public int _state
    cdef public dict initial_state_distribution
    cdef map[int, double] _transition_map
    cpdef transitions(self, int state, int action)
    cdef map[int, double] transitions_cy(self, int state, int action)
    cpdef double reward(self, int state, int action, int next_state)
    cpdef observation(self, int state)
    cpdef step(self, int action)
    cpdef step_state(self, int action)
    cpdef reset(self)
    cpdef int reset_state(self)
    cpdef transition_matrix(self)
    cpdef reward_matrix(self)
    cpdef set_state(self, int state)
    cpdef int get_state(self)

cdef class LinkedListEnv(TabularEnv):
    cdef double transition_noise
