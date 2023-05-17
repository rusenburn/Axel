import gym
import time
import torch as T
import numpy as np
from typing import Callable,Sequence
from gym.vector import AsyncVectorEnv
from collections import deque
from torch.nn.utils.clip_grad import clip_grad_norm_

from axel.common.networks import ActorNetwork, CnnActorNetwork, CnnCriticNetwork, CriticNetwork, ImpalaActorCritic, ImpalaCritic
from axel.common.running_normalizer import RunningNormalizer
from axel.common.utils import calculate_explained_variance, seconds_to_HMS

class Impala():
    def __init__(
            self,
            env_fns:Sequence[Callable[[],gym.Env]],
            total_steps=1_000_000,
            step_size=20,
            batch_size=32,
            replay_size=512,
            critic_coef=0.5,
            entropy_coef=0.01,
            gamma=0.99,
            c_=1,
            rho=1,
            lr=2e-4,
            max_grad_norm=40,
            normalize_adv=True,
            normalize_rewards=True,
            max_reward_norm=3,
            decay_lr=True,
            residual=True
            ) -> None:
        
        self.vec_env = AsyncVectorEnv(env_fns)

        self.n_actors = len(env_fns)
        self.n_game_actions = self.vec_env.single_action_space.n
        self.observations_shape= self.vec_env.single_observation_space.shape

        self.total_steps = int(total_steps)
        self.step_size = int(step_size)

        self.gamma = gamma
        self.c_ = c_
        self.rho = rho

        self.batch_size = int(batch_size)
        self.replay_size = int(replay_size)
        self.lr = lr
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        self.decay_lr = decay_lr
        self.max_grad_norm =max_grad_norm

        self.normalize_adv = normalize_adv
        self.normalize_rewards = normalize_rewards
        self.max_reward_norm = max_reward_norm
        self.rewards_normalizer = RunningNormalizer(maxlen=1000 * self.step_size,gamma=gamma)


        if residual:
            self.actor:ActorNetwork = ImpalaActorCritic(self.observations_shape,self.n_game_actions)
            self.critic:CriticNetwork = ImpalaCritic(self.observations_shape)
        else:
            self.actor:ActorNetwork = CnnActorNetwork(self.observations_shape,self.n_game_actions)
            self.critic:CriticNetwork = CnnCriticNetwork(self.observations_shape)

        self.actor_optim = T.optim.RMSprop(self.actor.parameters(),lr=self.lr,alpha=0.99,eps=1e-5)
        self.critic_optim = T.optim.RMSprop(self.critic.parameters(),lr=self.lr,alpha=0.99,eps=1e-5)

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)
    

    def run(self):
        t_start = time.perf_counter()
        device = self.device
        observations : np.ndarray
        infos : dict
        scaler = 1
        log_interval = 1000
        next_log = log_interval

        scores = deque([],maxlen=100)
        lengths = deque([],maxlen=100)
        policy_losses = []
        critic_losses = []
        entropies = []
        explained_variances = []

        observations,infos = self.vec_env.reset()
        
        current_step = 0
        assert self.replay_size % self.n_actors == 0
        replay_max_steps = self.replay_size // self.n_actors
        all_actor_log_probs : deque[np.ndarray] = deque(maxlen=replay_max_steps)
        all_actions : deque[np.ndarray] = deque(maxlen=replay_max_steps)
        all_observations : deque[np.ndarray] = deque(maxlen=replay_max_steps)
        all_rewards :deque[np.ndarray] = deque(maxlen=replay_max_steps)
        all_terminals : deque[np.ndarray] = deque(maxlen=replay_max_steps)
        while current_step < self.total_steps:
            
            # all_values : list[np.ndarray] = []

            for step in range(self.step_size):
                observation_t = T.tensor(observations , dtype=T.float32,device=device)
                with T.no_grad():
                    probs = self.actor.act(observation_t)
                    # values_t = self.critic.evaluate(observation_t)
                
                dist  = T.distributions.Categorical(probs)

                actions_t = dist.sample()
                actions_log_probs_t = dist.log_prob(actions_t)
                actions_log_probs = actions_log_probs_t.cpu().numpy()
                actions = actions_t.cpu().numpy()

                new_obs ,rewards,terminals,truncs,infos = self.vec_env.step(actions)
                current_step += self.n_actors
                if self.normalize_rewards:
                    rewards, scaler = self.rewards_normalizer.normalize_rewards(np.reshape(rewards,(1,-1)),np.reshape(terminals,(1,-1)),self.max_reward_norm)
                    rewards = rewards[0]

                ters = np.argwhere(terminals)

                for idx in ters:
                    t_score = infos["final_info"][idx[0]]["episode"]["r"]
                    t_length = infos["final_info"][idx[0]]["episode"]["l"]
                    scores.append(t_score)
                    lengths.append(t_length)

                terminals = np.logical_or(terminals,truncs)
                all_observations.append(observations)
                all_actions.append(actions)
                all_actor_log_probs.append(actions_log_probs)
                all_rewards.append(rewards)
                all_terminals.append(terminals)
                # all_values.append(values_t.squeeze().cpu().numpy())
                observations = new_obs
            
            if len(all_observations) * self.n_actors <self.batch_size:
                continue
            last_obs_t = T.tensor(observations,dtype=T.float32,device=device)

            with T.no_grad():
                last_values_t = self.critic.evaluate(last_obs_t)
            
            last_values_ar = last_values_t.squeeze().cpu().numpy()

            all_states_ar = np.array(all_observations,dtype=np.float32)
            all_actions_ar = np.array(all_actions,dtype=np.int32)
            all_terminals_ar = np.array(all_terminals,dtype=np.bool8)
            all_rewards_ar = np.array(all_rewards,dtype=np.float32)
            # all_values_ar = np.array(all_values,dtype=np.float32)

            n_examples = len(all_states_ar)
            all_actor_log_prob_ar = np.array(all_actor_log_probs,dtype=np.float32)

            all_states_t = T.tensor(all_states_ar,dtype=T.float32,device=device)
            
            with T.no_grad():
                shaped = all_states_t.reshape((-1,*self.observations_shape))
                probs = self.actor.act(shaped)
                all_values = self.critic.evaluate(shaped)
            
            all_actions_t = T.tensor(all_actions_ar.flatten(),dtype=T.int32,device=device)
            all_values_ar = all_values.squeeze().cpu().numpy().reshape((n_examples,self.n_actors))
            dist = T.distributions.Categorical(probs)
            learner_probs :T.Tensor = dist.log_prob(all_actions_t)
            all_learner_log_probs_ar:np.ndarray = learner_probs.cpu().numpy().reshape((n_examples,self.n_actors))

            qs,vs,i_s = self.calculate_v_trace(all_rewards_ar,all_values_ar,all_learner_log_probs_ar,all_actor_log_prob_ar,all_terminals_ar,last_values_ar)

            explained_variance = calculate_explained_variance(all_values_ar,vs)

            policy_loss , critic_loss, entropy = self.train_network(
                all_states_ar,
                all_actions_ar,
                qs,
                vs,
                i_s
            )
            policy_losses.append(policy_loss)
            critic_losses.append(critic_loss)
            entropies.append(entropy)
            explained_variances.append(explained_variance)

            if current_step >= next_log:
                next_log= next_log + log_interval
                duration = time.perf_counter() - t_start
                hours,minutes,seconds = seconds_to_HMS(duration)

                print("*" * 20)
                print(f"Steps: {current_step} of {self.total_steps} {current_step*100/self.total_steps:0.1f}%")
                print(f"Duration: {hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}")
                print(f"FPS: {int(current_step/duration)}")
                print(f"Scores Mean: {np.mean(scores):0.3f}")
                print(f"Lengths Mean: {np.mean(lengths):0.3f}")
                print(f"Explained variance: {np.mean(explained_variances):0.3f}")
                print(f"Policy Loss: {np.mean(policy_losses):0.3f}")
                print(f"Critic Loss: {np.mean(critic_losses):0.3f}")
                print(f"Entropy: {np.mean(entropies):0.3f}")
                print(f"Scaler: {scaler:0.3f}")

                explained_variances.clear()
                policy_losses.clear()
                critic_losses.clear()
                entropies.clear()
                

    def train_network(self,all_states:np.ndarray,all_actions:np.ndarray,all_qs:np.ndarray,all_vs:np.ndarray,all_is:np.ndarray):
        n_steps = len(all_states)
        n_examples = n_steps*self.n_actors
        all_states = all_states.reshape((n_examples,*self.observations_shape))
        all_actions = all_actions.reshape((n_examples,))
        all_qs = all_qs.reshape((n_examples,))
        all_vs = all_vs.reshape((n_examples,))
        all_is = all_is.reshape((n_examples,))

        sample_size = min(self.replay_size // 2 , n_examples)
        sample_size = sample_size //self.batch_size * self.batch_size
        assert sample_size % self.batch_size == 0
        n_batches = sample_size // self.batch_size

        actor_losses = []
        critic_losses = []
        all_entropies = []
        total_losses = []
        
        for i in range(n_batches):
            batch =  np.random.choice(n_examples,size=self.batch_size,replace=False)
            
            states_batch = T.tensor(all_states[batch],dtype=T.float32,device=self.device)
            actions_batch = T.tensor(all_actions[batch],dtype=T.int32,device=self.device)
            qs_batch = T.tensor(all_qs[batch],dtype=T.float32,device=self.device)
            vs_batch = T.tensor(all_vs[batch],dtype=T.float32,device=self.device)
            is_batch = T.tensor(all_is[batch],dtype=T.float32,device=self.device)


            probs = self.actor.act(states_batch)
            values = self.critic.evaluate(states_batch)
            values = values.squeeze()
            dist = T.distributions.Categorical(probs)
            entropy = T.mean(dist.entropy())
            log_probs = dist.log_prob(actions_batch)
            actor_loss =  -T.mean(is_batch * log_probs*(qs_batch-values.detach()))

            critic_loss = T.mean(0.5*(vs_batch-values)**2)
            total_loss = actor_loss - self.entropy_coef* entropy + self.critic_coef*critic_loss

            total_losses.append(total_loss)
            
            actor_losses.append(actor_loss.detach().cpu().item())
            critic_losses.append(critic_loss.detach().cpu().item())
            all_entropies.append(entropy.detach().cpu().item())

        if self.max_grad_norm:
            clip_grad_norm_(self.actor.parameters(),self.max_grad_norm)
            clip_grad_norm_(self.critic.parameters(),self.max_grad_norm)
        
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        total_loss = T.stack(total_losses,dim=0).sum()
        total_loss.backward()
        self.actor_optim.step()
        self.critic_optim.step()
        return np.mean(actor_losses) , np.mean(critic_losses) , np.mean(all_entropies)
        

    def calculate_v_trace(self,all_rewards:np.ndarray,all_values:np.ndarray,all_learner_log_probs:np.ndarray,all_actor_log_probs:np.ndarray,all_terminals:np.ndarray,last_values_ar:np.ndarray):
        n_steps = len(all_rewards)
        next_values = last_values_ar.copy()
        next_vs = last_values_ar.copy()
        all_ratios = np.exp(all_learner_log_probs - all_actor_log_probs)
        qs = np.zeros_like(all_values)
        vs = np.zeros_like(all_values)
        i_s = np.zeros_like(all_values)
        for step in reversed(range(n_steps)):
            current_reward = all_rewards[step]
            current_terminals = all_terminals[step]
            current_ratio = all_ratios[step]
            current_values = all_values[step]
            next_values[current_terminals] = 0
            next_vs[current_terminals] = 0
            rho_t = np.full_like(current_ratio,self.rho)
            predicate = current_ratio<self.rho
            rho_t[predicate] = current_ratio[predicate]
            
            i_s[step] = rho_t
            c_t = np.full_like(current_ratio,self.c_)
            predicate = current_ratio < self.c_
            c_t[predicate] = current_ratio[predicate]
            # check delta
            delta = rho_t * (current_reward + self.gamma * next_values - current_values)
            current_vs = current_values + delta + self.gamma*c_t*(next_vs - next_values)
            current_qs = current_reward + self.gamma * next_vs

            vs[step] = current_vs
            qs[step] = current_qs

            next_vs = np.copy(current_vs)
            next_values = np.copy(current_values)
        
        return qs,vs,i_s

        
    






