import gym
import os
import time
import numpy as np
import torch as T

from typing import Callable,Sequence
from gym.vector.async_vector_env import AsyncVectorEnv
from collections import deque
from torch.nn.utils.clip_grad import clip_grad_norm_
from axel.common.networks import ActorNetwork, CnnActorNetwork, CnnCriticNetwork, CriticNetwork, ImpalaActorCritic, ImpalaCritic
from axel.common.running_normalizer import RunningNormalizer
from axel.common.utils import calculate_explained_variance, seconds_to_HMS

class A2C():
    def __init__(self,
                 env_fns:Sequence[Callable[[],gym.Env]],
                 total_steps=1_000_000,
                 step_size=5,
                 critic_coef=0.5,
                 entropy_coef=.01,
                 lr=2e-4,
                 max_grad_norm=0.5,
                 gamma=0.99,
                 normalize_adv=False,
                 normalize_rewards=True,
                 max_reward_norm=3,
                 decay_lr=True,
                 residual=True
                 ) -> None:
        
        self.vec_env = AsyncVectorEnv(env_fns)

        self.n_actors = len(env_fns)
        self.n_game_actions = self.vec_env.single_action_space.n
        self.observation_shape = self.vec_env.single_observation_space.shape


        self.total_steps  = int(total_steps)   
        self.step_size = int(step_size)

        self.gamma = gamma
        self.lr = lr
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        self.decay_lr = decay_lr
        self.normalize_adv = normalize_adv
        self.normalize_rewards = normalize_rewards
        self.max_reward_norm = max_reward_norm
        self.max_grad_norm = max_grad_norm
        self.reward_normalizer = RunningNormalizer(maxlen=1000,gamma=gamma)


        if residual:
            self.actor: ActorNetwork = ImpalaActorCritic(self.observation_shape,self.n_actors)
            self.critic:CriticNetwork = ImpalaCritic(self.observation_shape)
        else:
            self.actor:ActorNetwork = CnnActorNetwork(self.observation_shape,self.n_actors)
            self.critic:CriticNetwork = CnnCriticNetwork(self.observation_shape)
        
        self.actor_optim = T.optim.RMSprop(self.actor.parameters(),lr=self.lr,alpha=0.99,eps=1e-5)
        self.critic_optim = T.optim.RMSprop(self.critic.parameters(),lr=self.lr,alpha=0.99,eps=1e-5)

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)

    
    def run(self):
        t_start = time.perf_counter()
        device = self.device
        observations :np.ndarray
        infos : dict
        scaler = 1
        log_interval = 1000
        next_log = log_interval

        scores = deque([],maxlen=100)
        lengths = deque([],maxlen=100)
        actor_losses = []
        critic_losses = []
        entropies = []
        explained_variances = []

        observations,infos = self.vec_env.reset()
        
        current_step = 0


        while current_step < self.total_steps:
            # run 1 training
            # all_log_probs : list[np.ndarray] = []
            all_actions : list[np.ndarray] = []
            all_observations : list[np.ndarray] = []
            all_rewards :list[np.ndarray] = []
            all_terminals : list[np.ndarray] = []
            all_values : list[np.ndarray] = []

            for step in range(self.step_size):
                observations_t = T.tensor(observations,dtype=T.float32,device=device)
                with T.no_grad():
                    probs = self.actor.act(observations_t)
                    values_t = self.critic.evaluate(observations_t)
                dist = T.distributions.Categorical(probs)
                actions_t = dist.sample()
                # action_log_probs = dist.log_prob(actions_t)
                actions = actions_t.cpu().numpy()

                new_obs, rewards,terminals,truncs,infos = self.vec_env.step(actions)
                current_step += self.n_actors

                ters = np.argwhere(terminals)
                for idx in ters:
                    t_score = infos["final_info"][idx[0]]["episode"]["r"]
                    t_length = infos["final_info"][idx[0]]["episode"]["l"]
                    scores.append(t_score)
                    lengths.append(t_length)
                terminals = np.logical_or(terminals,truncs)
                all_observations.append(observations)
                all_actions.append(actions)
                # all_log_probs.append(action_log_probs)
                all_rewards.append(rewards)
                all_terminals.append(terminals)
                all_values.append(values_t.squeeze().cpu().numpy())
                observations = new_obs

            last_obs_t = T.tensor(observations,dtype=T.float32,device=device)
            with T.no_grad():
                last_values_t = self.critic.evaluate(last_obs_t)
            last_values_ar= last_values_t.squeeze().cpu().numpy()

            all_states_ar = np.array(all_observations,dtype=np.float32)
            all_actions_ar = np.array(all_actions,dtype=np.int32)
            all_terminals_ar = np.array(all_terminals,dtype=np.bool8)
            all_rewards_ar = np.array(all_rewards,dtype=np.float32)
            all_values_ar = np.array(all_values,dtype=np.float32)


            if self.normalize_rewards:
                all_rewards_ar,scaler = self.reward_normalizer.normalize_rewards(all_rewards_ar,all_terminals_ar,self.max_reward_norm)
                
            
            all_adv_ar ,all_returns_ar = self.calculate_adv(all_rewards_ar,all_values_ar,all_terminals_ar,last_values_ar)

            explained_variance = calculate_explained_variance(all_values_ar,all_returns_ar)
            explained_variances.append(explained_variance)

            if self.decay_lr:
                decay_rate = 0.1
                decay =  decay_rate ** (current_step / self.total_steps)
                lr = self.lr * decay
                for g in self.actor_optim.param_groups:
                    g['lr'] = lr
                for g in self.critic_optim.param_groups:
                    g['lr'] = lr

            actor_loss , critic_loss , entropy = self.train_networks(all_states_ar,all_actions_ar,all_adv_ar,all_returns_ar)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            entropies.append(entropy)


            if current_step > next_log:
                next_log = next_log + log_interval
                duration = time.perf_counter() - t_start
                hours,minutes,seconds = seconds_to_HMS(duration)

                print("*" * 20)
                print(f"Steps: {current_step} of {self.total_steps} {current_step*100/self.total_steps:0.1f}%")
                print(f"Duration: {hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}")
                print(f"FPS: {int(current_step/duration)}")
                print(f"Scores Mean: {np.mean(scores):0.3f}")
                print(f"Lengths Mean: {np.mean(lengths):0.3f}")
                print(f"Explained variance: {np.mean(explained_variances):0.3f}")
                print(f"Actor Loss: {np.mean(actor_losses):0.3f}")
                print(f"Critic Loss: {np.mean(critic_losses):0.3f}")
                print(f"Entropy: {np.mean(entropies):0.3f}")
                print(f"Scaler: {scaler:0.3f}")

                explained_variances.clear()
                actor_losses.clear()
                critic_losses.clear()
                entropies.clear()


    def train_networks(self,all_state_ar:np.ndarray,all_actions_ar:np.ndarray,all_adv_ar:np.ndarray,all_returns_ar:np.ndarray):
        sample_size = self.n_actors * self.step_size
        states_t = T.tensor(all_state_ar,dtype=T.float32,device=self.device).reshape((sample_size,*self.observation_shape))
        all_action_t = T.tensor(all_actions_ar,dtype=T.int32,device=self.device).flatten()
        all_adv_t = T.tensor(all_adv_ar,dtype=T.float32,device=self.device).flatten()
        all_returns_t = T.tensor(all_returns_ar,dtype=T.float32,device=self.device).flatten()

        if self.normalize_adv:
            all_adv_t = (all_adv_t - all_adv_t.mean()) / (all_adv_t.std()+1e-8)
        
        probs = self.actor.act(states_t)
        dist = T.distributions.Categorical(probs)
        entropy = T.mean(dist.entropy())

        log_probs :T.Tensor= dist.log_prob(all_action_t)
        actor_loss = T.mean(-log_probs * all_adv_t)
        
        values = self.critic.evaluate(states_t).squeeze()
        critic_loss = T.mean(0.5*(values - all_returns_t)**2)

        total_loss = actor_loss +  self.critic_coef*critic_loss - self.entropy_coef * entropy

        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()

        total_loss.backward()
        
        if self.max_grad_norm:
            clip_grad_norm_(self.actor.parameters(),self.max_grad_norm)
            clip_grad_norm_(self.critic.parameters(),self.max_grad_norm)
        
        self.actor_optim.step()
        self.critic_optim.step()

        return actor_loss.detach().cpu().item(),critic_loss.detach().cpu().item(),entropy.detach().cpu().item()

    def calculate_adv(self,all_rewards:np.ndarray,all_values:np.ndarray,all_terminals:np.ndarray,last_values:np.ndarray):
        assert all_values.ndim == 2
        n_steps,n_actors = all_values.shape
        returns = np.zeros_like(all_values)
        advs = np.zeros_like(all_values)
        next_values = last_values.copy()
        for step in reversed(range(n_steps)):
            current_rewards : np.ndarray = all_rewards[step]
            current_values : np.ndarray = all_values[step]
            current_terminals : np.ndarray= all_terminals[step]
            next_values[current_terminals] = 0

            current_returns : np.ndarray= current_rewards + self.gamma * next_values
            current_advs : np.ndarray= current_returns - current_values

            returns[step] = current_returns
            advs[step] = current_advs
            next_values = current_returns.copy()
        
        return advs,returns

            
            



            
