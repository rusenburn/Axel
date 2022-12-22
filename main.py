import gym
from gym.vector import SyncVectorEnv, AsyncVectorEnv
import numpy as np
import time
from axel.ppo.ppo import Ppo
from axel.pop3d.pop3d import Pop3d
from axel.ppg.ppg import Ppg
from axel.common.env_wrappers import apply_wrappers


def get_env():
    return apply_wrappers(env=gym.make(id="ALE/Riverraid-v5"))


def train_pop3d():
    env_fns = [get_env for _ in range(8)]
    vec_env = AsyncVectorEnv(env_fns=env_fns)
    pop3d = Pop3d(vec_env=vec_env, step_size=128, total_steps=int(1e5),beta=10)
    pop3d.run()
    vec_env.close()

def train_ppo():
    env_fns = [get_env for _ in range(8)]
    vec_env = AsyncVectorEnv(env_fns=env_fns)
    ppo = Ppo(vec_env=vec_env, step_size=128, total_steps=int(1e5),policy_clip=0.1)
    ppo.run()
    vec_env.close()


def train_ppg():
    env_fns = [get_env for _ in range(8)]
    vec_env = AsyncVectorEnv(env_fns=env_fns)
    ppg = Ppg(
        vec_env=vec_env,total_steps=int(1e6),
        step_size=256,
        clip_ratio=0.2,n_pi=16,
        n_aux_epochs=6,
        n_pi_epochs=1,n_v_epochs=1,
        n_batches=8,n_aux_batches=16,
        lr=5e-4)
    ppg.run()
    vec_env.close()
def main():
    train_pop3d()
    # train_ppg()
    # train_ppo()




if __name__ == "__main__":
    main()
