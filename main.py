import gym
from gym.vector import SyncVectorEnv,AsyncVectorEnv
import numpy as np
import time
from axel.ppo.ppo import Ppo
from axel.common.env_wrappers import apply_wrappers


def get_env():
    return apply_wrappers(env=gym.make(id="ALE/Riverraid-v5"))
def main():
    env_fns = [get_env for _ in range(8)]
    vec_env = AsyncVectorEnv(env_fns=env_fns)
    ppo = Ppo(vec_env=vec_env,step_size=128)
    ppo.run()

if __name__ == "__main__":
    main()