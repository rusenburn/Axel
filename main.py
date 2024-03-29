import gym
from gym.vector import SyncVectorEnv, AsyncVectorEnv
import numpy as np
import time
from axel.a2c.a2c import A2C
from axel.impala.impala import Impala
from axel.ppo.ppo import Ppo
from axel.pop3d.pop3d import Pop3d
from axel.ppg.ppg import Ppg
from axel.common.env_wrappers import apply_wrappers,apply_wrappers_2
from axel.ppo.recurrent_ppo import RecurrentPPO
from axel.trainer_builder import TrainerBuilder
from axel.dddq.dddq import DDDQ


def get_env():
    # return apply_wrappers(env=gym.make(id="ALE/Riverraid-v5"))
    return apply_wrappers(env=gym.make("CarRacing-v2", continuous=False))

def get_env_fn(env_name:str,skip=4,grayscale=True,resize=84,framestack=1,reward_scale=1/100):
    def custom_env():
        return apply_wrappers(env=gym.make(id=env_name),skip=skip,grayscale=grayscale,resize=resize,framestack=framestack,reward_scale=reward_scale)
    return custom_env

def get_cartpole():
    return gym.make("CartPole-v1")


def get_acrobat():
    return gym.make("Acrobot-v1")


def train_pop3d():
    env_fns = [get_env for _ in range(8)]
    vec_env = AsyncVectorEnv(env_fns=env_fns)
    pop3d = Pop3d(vec_env=vec_env, step_size=128,
                  total_steps=int(1e6), beta=10, normalize_rewards=True)
    pop3d.run()
    vec_env.close()


def train_ppo():
    env_fns = [get_env for _ in range(8)]
    vec_env = AsyncVectorEnv(env_fns=env_fns)
    ppo = (TrainerBuilder
            .use_ppo()
            .total_steps(1e6)
            .policy_clip(0.1)
            .enable_rewards_normalisation()
            .disable_advantages_normalisation()
            .vec_env(vec_env)
            .build())
    ppo.run()
    vec_env.close()


def train_ppg():
    env_fns = [get_env for _ in range(8)]
    vec_env = AsyncVectorEnv(env_fns=env_fns)
    ppg = (TrainerBuilder.use_ppg()
                .step_size(256)
                .clip_ratio(0.2)
                .n_pi(16)
                .n_aux_epochs(6)
                .n_pi_epochs(1)
                .n_v_epochs(1)
                .n_batches(8)
                .n_aux_batches(16)
                .learning_rate(5e-4)
                .enable_reward_normalization(3)
                .gamma(0.99)
                .disable_lr_decay()
                .vec_env(vec_env)
                .disable_advantage_normalization()
                .entropy_coef(0)
                .build())
    ppg.run()
    vec_env.close()


def train_a2c():
    env_fns = [get_env for _ in range(8)]
    a2c = A2C(env_fns)
    a2c.run()


def train_impala():
    env_fns = [get_env for _ in range(8)]
    impala = Impala(env_fns)
    impala.run()
def train_ppg_cartpole():
    env_fns = [get_cartpole for _ in range(8)]
    vec_env = AsyncVectorEnv(env_fns=env_fns)
    ppg = Ppg(
        vec_env=vec_env, total_steps=int(1e6),
        step_size=20,
        clip_ratio=0.2,
        n_pi=32,
        n_aux_epochs=6,
        n_pi_epochs=1, n_v_epochs=1,
        n_batches=4,
        n_aux_batches=8,
        lr=3e-4,
        normalize_adv=False,
        entropy_coef=0.00,
        normalize_rewards=True,
        max_grad_norm=1)
    ppg.run()


def train_ppg_acrobat():
    env_fns = [get_acrobat for _ in range(8)]
    vec_env = AsyncVectorEnv(env_fns=env_fns)
    ppg = Ppg(
        vec_env=vec_env, total_steps=int(1e6),
        step_size=20,
        clip_ratio=0.2,
        n_pi=32,
        n_aux_epochs=6,
        n_pi_epochs=1, n_v_epochs=1,
        n_batches=4,
        n_aux_batches=8,
        lr=3e-4,
        normalize_adv=True,
        entropy_coef=0.00,
        normalize_rewards=True,
        max_grad_norm=1)
    ppg.run()


def train_ppo_catrpole():
    env_fns = [get_cartpole for _ in range(8)]
    vec_env = AsyncVectorEnv(env_fns=env_fns)
    # ppo = Ppo(vec_env=vec_env,
    #           step_size=20,
    #           total_steps=int(2e6),
    #           policy_clip=0.1,
    #           entropy_coef=0,
    #           normalize_rewards=False,
    #           normalize_adv=True,
    #           )
    ppo = (TrainerBuilder.use_ppo()
                .use_defaults()
                .vec_env(vec_env)
                .step_size(20)
                .total_steps(2e6)
                .policy_clip(0.1)
                .entropy_coef(0)
                .disable_rewards_normalisation()
                .enable_advantages_normalisation()
                .build())
    ppo.run()

def train_pop3d_cartpole():
    env_fns = [get_cartpole for _ in range(16)]
    vec_env = SyncVectorEnv(env_fns=env_fns)
    pop3d = Pop3d(vec_env,
            total_steps=1e6,
            step_size=20,
            entropy_coef=0,
            beta=10,
            normalize_adv=True
            )
    pop3d.run()


def train_recurrent_ppo():
    envs_fns = [get_env_fn("ALE/Riverraid-v5",reward_scale=1/100,framestack=1) for _ in range(8)]
    ppo = RecurrentPPO(total_steps=8_000_000,game_fns=envs_fns,gamma=0.99,normalize_adv=False,decay_lr=False)
    ppo.run()

def train_dddq():
    def env_fn():
        return apply_wrappers_2(gym.make(id="ALE/Riverraid-v5"),framestack=4)
    
    dddq = DDDQ(env_fns= [env_fn for _ in range(8)],name="RiverFrameStack4")
    dddq.run()
def main():
    # train_pop3d()
    # train_recurrent_ppo()
    # train_impala()
    # train_a2c()
    # train_ppg()
    # train_ppo()
    # train_ppg_cartpole()
    # train_ppg_acrobat()
    # train_ppo_catrpole()
    # train_pop3d_cartpole()
    train_dddq()


if __name__ == "__main__":
    main()
