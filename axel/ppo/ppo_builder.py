import gym
from .ppo import Ppo
from axel.common.env_wrappers import apply_wrappers
from gym.vector import AsyncVectorEnv,SyncVectorEnv,VectorEnv
class PpoBuilder():
    def __init__(self) -> None:
        self.use_defaults()
    def build(self):
        return Ppo(
            vec_env=self._vec_env,
            total_steps=self._total_steps,
            step_size=self._step_size,
            n_batches=self._n_batches,
            n_epochs=self._n_epochs,
            gamma=self._gamma,
            gae_lam=self._gae_lam,
            policy_clip=self._policy_clip,
            lr=self._lr,
            entropy_coef=self._entropy_coef,
            critic_coef=self._critic_coef,
            max_grad_norm=self._max_grad_norm,
            normalize_adv=self._norm_advantages,
            normalize_rewards=self._norm_rewards,
            max_reward_norm=self._max_rew_norm,
            decay_lr=self._decay_lr
            )
    
    def use_defaults(self):
        def get_env():
            return apply_wrappers(env=gym.make(id="ALE/Riverraid-v5"))
        env_fns = [get_env for _ in range(8)]
        self._vec_env = AsyncVectorEnv(env_fns=env_fns)
        self._total_steps=1_000_000
        self._step_size = 128
        self._n_batches = 4
        self._n_epochs = 4
        self._gamma = 0.99
        self._gae_lam = 0.95
        self._policy_clip = 0.2
        self._lr = 2.5e-4
        self._entropy_coef = 0.01
        self._critic_coef = 0.5
        self._max_grad_norm = 0.5
        self._norm_advantages = False
        self._norm_rewards = True
        self._max_rew_norm = 3
        self._decay_lr = True
        return self

    def vec_env(self,vec_env:gym.vector.VectorEnv)->'PpoBuilder':
        self._vec_env = vec_env
        return self
    
    def total_steps(self,total_steps:int):
        self._total_steps = int(total_steps)
        return self
    
    def step_size(self,step_size:int):
        self._step_size = int(step_size)
        return self
    
    def n_batches(self,n_batches:int):
        self._n_batches = int(n_batches)
        return self
    
    def n_epochs(self,n_epochs:int):
        self._n_epochs = int(n_epochs)
        return self
    
    def gamma(self,gamma:float):
        if gamma < 0 or gamma >1:
            raise ValueError("Gamma must be between 0 and 1 inclusive")
        self._gamma = gamma
        return self
    
    def gae_lambda(self,gae_lam:float):
        if gae_lam <0 or gae_lam > 1 :
            raise ValueError("Gae lam parameter must be between 0 and 1 inclusive")
        self._gae_lam = gae_lam
        return self
    
    def policy_clip(self,policy_clip:float):
        self._policy_clip = policy_clip
        return self
    
    def learning_rate(self,lr:float):
        if lr <0:
            raise ValueError("Learning rate should not be less than 0")
        self._lr = lr
        return self
    
    def entropy_coef(self,entropy_coef:float):
        self._entropy_coef = entropy_coef
        return self
    
    def critic_coef(self,critic_coef:float):
        self._critic_coef = critic_coef
        return self
    
    def enable_max_grad_norm(self,max_grad_norm:float=0.5):
        self._max_grad_norm = max_grad_norm
        return self
    
    def disable_max_grad_norm(self):
        self._max_grad_norm = None
        return self
    
    def enable_advantages_normalisation(self):
        self._norm_advantages = True
        return self
    
    def disable_advantages_normalisation(self):
        self._norm_advantages = False
        return self
    
    def enable_rewards_normalisation(self,max_rewards_norm:float=3):
        self._norm_rewards = True
        self._max_rew_norm = max_rewards_norm
        return self
    
    def disable_rewards_normalisation(self):
        self._norm_rewards = False
        return self

    def enable_lr_decay(self,is_enabled=True):
        self._decay_lr = is_enabled
        return self
    
    def disable_lr_decay(self):
        self._decay_lr = False
        return self
    
