import gym
from gym.vector import SyncVectorEnv, AsyncVectorEnv
import numpy as np
import time
from axel.ppo.ppo import Ppo
from axel.pop3d.pop3d import Pop3d
from axel.common.env_wrappers import apply_wrappers


def get_env():
    return apply_wrappers(env=gym.make(id="ALE/Riverraid-v5"))


def train_pop3d():
    env_fns = [get_env for _ in range(8)]
    vec_env = AsyncVectorEnv(env_fns=env_fns)
    pop3d = Pop3d(vec_env=vec_env, step_size=128, total_steps=int(1e6),beta=10)
    pop3d.run()

def train_ppo():
    env_fns = [get_env for _ in range(8)]
    vec_env = AsyncVectorEnv(env_fns=env_fns)
    ppo = Ppo(vec_env=vec_env, step_size=128, total_steps=int(1e6),policy_clip=0.1)
    ppo.run()

def main():
    train_pop3d()




if __name__ == "__main__":
    main()
