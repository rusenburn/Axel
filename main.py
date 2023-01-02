import gym
from gym.vector import SyncVectorEnv, AsyncVectorEnv
import numpy as np
import time
from axel.ppo.ppo import Ppo
from axel.pop3d.pop3d import Pop3d
from axel.ppg.ppg import Ppg
from axel.common.env_wrappers import apply_wrappers
from axel.trainer_builder import TrainerBuilder


def get_env():
    return apply_wrappers(env=gym.make(id="ALE/Riverraid-v5"))


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
    # ppo = Ppo(vec_env=vec_env, step_size=128,
    #           total_steps=int(1e6),
    #           policy_clip=0.1,
    #           normalize_rewards=True,
    #           normalize_adv=False)
    ppo = (TrainerBuilder
            .use_ppo()
            .use_defaults()
            .total_steps(1e6)
            .policy_clip(0.1)
            .enable_rewards_normalisation()
            .disable_advantages_normalisation()
            .build())
    ppo.run()
    vec_env.close()


def train_ppg():
    env_fns = [get_env for _ in range(8)]
    vec_env = AsyncVectorEnv(env_fns=env_fns)
    # ppg = Ppg(
    #     vec_env=vec_env, total_steps=int(1e6),
    #     step_size=256,
    #     clip_ratio=0.2,
    #     n_pi=32,
    #     n_aux_epochs=6,
    #     n_pi_epochs=1, n_v_epochs=1,
    #     n_batches=8,
    #     n_aux_batches=16,
    #     lr=5e-4,
    #     normalize_rewards=True,
    #     normalize_adv=False,
    #     entropy_coef=0)

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
                .disable_advantage_normalization()
                .entropy_coef(0)
                .build())
    ppg.run()
    vec_env.close()


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
            entropy_coef=0,beta=10
            )
    pop3d.run()

def main():
    # train_pop3d()
    train_ppg()
    # train_ppo()
    # train_ppg_cartpole()
    # train_ppg_acrobat()
    # train_ppo_catrpole()
    # train_pop3d_cartpole()


if __name__ == "__main__":
    main()
