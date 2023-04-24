from axel.common.env_wrappers import apply_wrappers
from .ppg import Ppg
import gym
import gym.vector
from gym.vector import AsyncVectorEnv

class PpgBuilder():
    def __init__(self) -> None:
        self.use_defaults()
    
    def build(self)->Ppg:
        return Ppg(
            vec_env=self._vec_env,
            total_steps=self._total_steps,
            step_size=self._step_size,
            n_pi=self._n_pi,
            n_pi_epochs=self._n_pi_epochs,
            n_v_epochs=self._n_v_epochs,
            n_aux_epochs=self._n_aux_epochs,
            beta_clone=self._beta_clone,
            n_aux_batches=self._n_aux_batches,
            gamma=self._gamma,
            gae_lam=self._gae_lam,
            n_batches=self._n_batches,
            clip_ratio=self._clip_ratio,
            lr=self._lr,
            entropy_coef=self._entropy_coef,
            max_grad_norm=self._max_grad_norm,
            normalize_adv=self._normalize_adv,
            normalize_rewards=self._normalize_rewards,
            max_reward_norm=self._max_rew_norm,
            decay_lr=self._decay_lr)

    def use_defaults(self)->'PpgBuilder':
        def get_env():
            return apply_wrappers(env=gym.make(id="ALE/Riverraid-v5"))
        env_fns = [get_env for _ in range(8)]
        self._vec_env = AsyncVectorEnv(env_fns=env_fns)
        self._total_steps = 1_000_000
        self._step_size = 256
        self._n_pi = 32
        self._n_pi_epochs = 1
        self._n_v_epochs = 1
        self._n_aux_epochs = 6
        self._beta_clone = 1
        self._n_aux_batches = 16
        self._gamma = 0.99
        self._gae_lam = 0.95
        self._n_batches = 8
        self._clip_ratio = 0.2
        self._lr = 5e-4
        self._entropy_coef = 0
        self._max_grad_norm = 0.5
        self._normalize_adv = False
        self._normalize_rewards = True
        self._max_rew_norm = 3
        self._decay_lr = True

    def vec_env(self,vec_env:gym.vector.VectorEnv):
        if self._vec_env == vec_env:
            return self
        self._vec_env.close()
        self._vec_env = vec_env
        return self
    
    def total_steps(self,total_steps:int):
        self._total_steps = int(total_steps)
        return self
    def step_size(self,step_size:int):
        self._step_size = int(step_size)
        return self
    
    def n_pi(self,n_pi:int):
        self._n_pi = int(n_pi)
        return self
    
    def n_pi_epochs(self,n_pi_epochs:int):
        self._n_pi_epochs = int(n_pi_epochs)
        return self
    
    def n_v_epochs(self,n_v_epochs:int):
        self._n_v_epochs = int(n_v_epochs)
        return self
    
    def n_aux_epochs(self,n_aux_epochs:int):
        self._n_aux_epochs = int(n_aux_epochs)
        return self
    
    def beta_clone(self,beta_clone:float):
        self._beta_clone = beta_clone
        return self
    
    def n_aux_batches(self,n_aux_batches:int):
        self._n_aux_batches = int(n_aux_batches)
        return self
    
    def gamma(self,gamma:float):
        self._gamma = gamma
        return self
    
    def gae_lam(self,gae_lam:float):
        self._gae_lam = gae_lam
        return self
    
    def n_batches(self,n_batches:int):
        self._n_batches = int(n_batches)
        return self
    
    def clip_ratio(self,clip_ratio:float):
        self._clip_ratio = clip_ratio
        return self
    
    def learning_rate(self,lr:float):
        self._lr = lr
        return self
    
    def entropy_coef(self,entropy_coef:float):
        self._entropy_coef = entropy_coef
        return self
    
    def enable_max_grad_norm(self,max_grad_norm:float):
        self._max_grad_norm = max_grad_norm
        return self
    
    def disable_max_grad_norm(self):
        self._max_grad_norm = None
        return self
    
    def enable_advantage_normalization(self,is_enabled=True):
        self._normalize_adv = is_enabled
        return self
    
    def enable_reward_normalization(self,max_reward_norm:float):
        self._normalize_rewards = True
        self._max_rew_norm = max_reward_norm
        return self
    
    def disable_advantage_normalization(self):
        self._normalize_adv = False
        return self
    
    def disable_reward_normalization(self):
        self._normalize_rewards = False
        return self
    
    def enable_lr_decay(self,is_enabled=True):
        self._decay_lr = is_enabled
        return self
    
    def disable_lr_decay(self):
        self._decay_lr = False
        return self

    



        