from lqrenv import *
from lqr_solver import *

if __name__ == '__main__':
    env = PointmassEnvTorque(initial_pos=np.array([0.1,0.1]), goal_pos=np.array([0.64, 0.76]))
    #env = PointmassEnvVelocity(initial_pos=np.array([0.1,0.1]), goal_pos=np.array([0.6, 0.6]))
    K, k, V, v, Q, q = solve_lqr_env(env, 100)

    obs = env.reset()
    for t in range(100):
        action = K[t].dot(obs) + k[t]
        print(obs, action)
        obs, reward, done, infos= env.step(action)
        print(reward)

