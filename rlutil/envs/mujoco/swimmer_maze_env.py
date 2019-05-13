from rlutil.envs.mujoco.maze_env import MazeEnv
from rlutil.envs.mujoco import swimmer

class SwimmerMazeEnv(MazeEnv):

    MODEL_FILE = 'swimmer.xml'
    MODEL_CLASS = swimmer.SwimmerEnv
    ORI_IND = 2

    MAZE_HEIGHT = 0.5
    MAZE_SIZE_SCALING = 4
    MAZE_MAKE_CONTACTS = True


if __name__ == "__main__":
    env = SwimmerMazeEnv()
    env.reset()

    for _ in range(1000):
        env.step(env.action_space.sample())
        env.render()
