import os
import time
import gym
import tqdm
import torch as T
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from torch.nn.utils.clip_grad import clip_grad_norm_
from ..common.networks import ActorCriticNetwork, CriticNetwork
from ..ppo.memory_buffer import MemoryBuffer


class Ppg():
    def __init__(self, vec_env: gym.vector.VectorEnv,
                 total_steps: int,
                 step_size=128,
                 n_pi=32,
                 n_pi_epochs=1,
                 n_v_epochs=1,
                 n_aux_epochs=6,
                 beta_clone=1,
                 n_aux_batches=8,
                 gamma=0.99,
                 gae_lam=0.95,
                 n_batches=4,
                 clip_ratio=0.2,
                 lr=2.5e-4
                 ) -> None:

        self.vec_env = vec_env
        self.n_workers = vec_env.num_envs
        self.n_game_action = vec_env.single_action_space.n
        self.observation_shape = vec_env.single_observation_space.shape

        self.total_steps = total_steps
        self.step_size = step_size

        # PPG hyperparameters
        self.n_pi = n_pi
        self.n_pi_epochs = n_pi_epochs
        self.n_v_epochs = n_v_epochs
        self.n_aux_epochs = n_aux_epochs
        self.beta_clone = beta_clone
        self.n_aux_batches = n_aux_batches

        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lam = gae_lam
        self.n_batches = n_batches
        self.clip_ratio = clip_ratio
        # self.lr = lr
        self.max_grad_norm = 0.5

        # networks
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.policy_network = ActorCriticNetwork(
            self.observation_shape, self.n_game_action)
        self.value_network = CriticNetwork(self.observation_shape)
        self.policy_network_optim = T.optim.Adam(
            self.policy_network.parameters(), lr=lr)
        self.value_network_optim = T.optim.Adam(
            self.value_network.parameters(), lr=lr)

        self.policy_network.to(device=self.device)
        self.value_network.to(device=self.device)

        self.memory_buffer = MemoryBuffer(
            self.observation_shape, self.n_workers, self.step_size)

    def run(self):
        # each phase consist of n_pi iterations which consist of n_workers * step_size steps
        # steps_per_phase = n_pi * n_workers * step_size
        steps_per_phase = self.n_pi * self.n_workers * self.step_size
        print(f"Steps per phase is {steps_per_phase}")
        n_phases = int(self.total_steps) // steps_per_phase
        def lr_fn(x): return (n_phases-x) / n_phases
        policy_network_scheduler = T.optim.lr_scheduler.LambdaLR(
            self.policy_network_optim, lr_fn)
        value_network_scheduler = T.optim.lr_scheduler.LambdaLR(
            self.value_network_optim, lr_fn)
        t_start = time.perf_counter()
        observations: np.ndarray
        infos: dict
        observations, infos = self.vec_env.reset()
        total_score = np.zeros((self.n_workers,), dtype=np.float32)
        s = deque([], 100)
        summary_time_step = []
        summary_duration = []
        summary_score = []
        summary_err = []
        for phase in range(1,n_phases+1):
            buffer = []
            for iteration in tqdm.tqdm(range(1,self.n_pi+1),f"Phase {phase} iterations:"):
                last_values_ar: np.ndarray
                for t in range(1,self.step_size+1):
                    actions, log_probs, values = self.choose_actions(
                        observations)
                    new_observations, rewards, dones, truncs, infos = self.vec_env.step(
                        actions)
                    self.memory_buffer.save(
                        observations, actions, log_probs, values, rewards, dones)
                    observations = new_observations
                    total_score += rewards
                    for i, d in enumerate(dones):
                        if d:
                            s.append(total_score[i])
                            total_score[i] = 0
                    if t == self.step_size:  # last step
                        last_values_t: T.Tensor
                        with T.no_grad():
                            last_values_t = self.value_network(
                                T.tensor(observations, dtype=T.float32, device=self.device))
                            last_values_ar = last_values_t.cpu().numpy()

                observation_sample, action_sample, log_probs_sample, value_sample, reward_sample, dones_sample = self.memory_buffer.sample()
                advantages_tensor = self._calculate_advantages(
                    reward_sample, value_sample, dones_sample, last_values_ar)

                observations_tensor = T.tensor(
                    observation_sample.reshape((self.n_workers*self.step_size,*self.observation_shape)), dtype=T.float32, device=self.device)
                actions_tensor = T.tensor(action_sample.reshape((self.n_workers*self.step_size,)), device=self.device)
                log_probs_tensor = T.tensor(
                    log_probs_sample.reshape((self.n_workers*self.step_size,)), dtype=T.float32, device=self.device)
                value_tensor = T.tensor(
                    value_sample.reshape((self.n_workers*self.step_size,)), dtype=T.float32, device=self.device)

                for ep in range(self.n_pi_epochs):
                    self.train_policy(observations_tensor, actions_tensor,
                                  log_probs_tensor, advantages_tensor)
                
                for ep in range(self.n_v_epochs):
                    self.train_value(observations_tensor,
                                 value_tensor + advantages_tensor,self.n_batches)
                buffer.append((observations_tensor.cpu().numpy(),value_tensor.cpu().numpy(),advantages_tensor.cpu().numpy()))
                self.memory_buffer.reset()

            all_obs_ar = np.concatenate([b[0] for b in buffer],axis=0 )
            returns_ar = np.concatenate([b[1]+b[2] for b in buffer],axis=0)
            all_obs_tensor = T.tensor(all_obs_ar,dtype=T.float32,device=self.device)
            returns_tensor = T.tensor(returns_ar,dtype=T.float32,device=self.device)
            buffer.clear()
            assert returns_tensor.device == self.device
            old_probs_tensor: T.Tensor
            with T.no_grad():
                old_probs_tensor, _ = self.policy_network(all_obs_tensor)
                old_probs_tensor = old_probs_tensor.detach()
            for ep in range(self.n_aux_epochs):
                self.train_aux(all_obs_tensor,old_probs_tensor, returns_tensor)
            policy_network_scheduler.step()
            value_network_scheduler.step()
            if T.cuda.is_available():
                T.cuda.empty_cache()

            if phase % 1 == 0:
                self.policy_network.save_model(os.path.join("tmp","ppg_policy.pt"))
                self.value_network.save_model(os.path.join("tmp","ppg_value.pt"))
            score_mean = np.mean(s)
            score_std = np.std(s)
            score_err = 1.95 * score_std / (len(s)**0.5)
            steps_done = phase * steps_per_phase
            total_duration = time.perf_counter()-t_start
            
            fps = steps_done // total_duration
            if phase % 1 == 0:
                print(f"*************************************")
                print(f"Phase:          {phase} of {n_phases}")
                print(f"learning_rate:  {policy_network_scheduler.get_last_lr()}")
                print(f"Total Steps:    {phase*steps_per_phase}")
                print(f"Total duration: {total_duration:0.2f} seconds")
                print(f"Average Score:  {score_mean:0.2f} ± {score_err:0.2f}")
                print(f"Average FPS:    {int(fps)}")

            summary_duration.append(total_duration)
            summary_time_step.append(phase*steps_per_phase)
            summary_score.append( score_mean)
            summary_err.append(score_err)
        

        self.plot_summary(summary_time_step,summary_score,summary_err,"Steps","Score","Step-Score.png")
        self.plot_summary(summary_duration,summary_score,summary_err,"Steps","Score","Duration-Score.png")

    def choose_actions(self,observation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        observation_tensor :T.Tensor = T.tensor(observation,dtype=T.float32,device=self.device)
        with T.no_grad():
            probs:T.Tensor
            probs,_ = self.policy_network(observation_tensor)
            values:T.Tensor = self.value_network(observation_tensor)
        
        action_sample = T.distributions.Categorical(probs)
        actions = action_sample.sample()
        log_probs :T.Tensor= action_sample.log_prob(actions)
        return actions.cpu().numpy() , log_probs.cpu().numpy(),values.squeeze(-1).cpu().numpy()


    def train_policy(self, observation: T.Tensor, actions: T.Tensor, log_probs: T.Tensor, advantages: T.Tensor):
        batch_size = (self.step_size*self.n_workers) // self.n_batches
        normalize_adv = True
    
        batch_starts = np.arange(0, self.n_workers*self.step_size,batch_size)
        indices = np.arange(self.n_workers*self.step_size)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_starts]

        for batch in batches:
            observation_batch = observation[batch]
            old_log_probs_batch = log_probs[batch]
            actions_batch = actions[batch]
            advantages_batch = advantages[batch].clone()
            if normalize_adv:
                advantages_batch = (advantages_batch - advantages_batch.mean()) / advantages_batch.std()
            
            predicted_probs:T.Tensor
            predicted_probs , _ = self.policy_network(observation_batch)
            probs_dist = T.distributions.Categorical(predicted_probs)
            entropy:T.Tensor = probs_dist.entropy().mean()
            predicted_log_probs:T.Tensor = probs_dist.log_prob(actions_batch)
            prob_ratio = (predicted_log_probs-old_log_probs_batch).exp()
            weighted_probs = advantages_batch * prob_ratio
            clipped_prob_ratio = prob_ratio.clamp(1-self.clip_ratio,1+self.clip_ratio)
            weighted_clipped_probs = advantages_batch * clipped_prob_ratio


            lclip = -T.min(weighted_probs,weighted_clipped_probs).mean()
            loss = lclip - 0.01*entropy

            self.policy_network.zero_grad()
            loss.backward()

            if self.max_grad_norm:
                clip_grad_norm_(self.policy_network.parameters(),max_norm=self.max_grad_norm)

            self.policy_network_optim.step()

    def train_value(self,observations:T.Tensor,returns:T.Tensor,n_batches:int):
        size = observations.shape[0]
        batch_size = size // n_batches
        batch_starts = np.arange(0, size,batch_size)
        indices = np.arange(size)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_starts]

        for batch in batches:
            observations_batch = observations[batch]
            predicted_values:T.Tensor = self.value_network(observations_batch)
            returns_batch :T.Tensor = returns[batch]
            predicted_values = predicted_values.squeeze()

            value_loss = 0.5 * (returns_batch - predicted_values )**2
            value_loss = value_loss.mean()

            self.value_network_optim.zero_grad()
            value_loss.backward()

            if self.max_grad_norm:
                clip_grad_norm_(self.value_network.parameters(),max_norm=self.max_grad_norm)
            self.value_network_optim.step()

    def train_aux(self,observations:T.Tensor,old_probs:T.Tensor,returns:T.Tensor):
        size = observations.shape[0]
        n_batches = self.n_aux_batches
        batch_size = size // n_batches
        batch_starts = np.arange(0, size,batch_size)
        indices = np.arange(size)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_starts]

        
        # Optimize Ljoint wrt theta_pi, on all data in buffer    
        for batch in batches:
            old_probs_batch = old_probs[batch]
            observations_batch = observations[batch]
            returns_batch = returns[batch]

            predicted_policy_values:T.Tensor
            new_probs: T.Tensor
            new_probs,predicted_policy_values = self.policy_network(observations_batch)
            predicted_policy_values = predicted_policy_values.squeeze(-1)

            aux_loss = 0.5 * ((returns_batch - predicted_policy_values)**2).mean()
            # kl_loss_fn = T.nn.KLDivLoss(reduction="batchmean")
            # kl_loss = kl_loss_fn(new_probs,old_probs_batch).mean()
            kl_loss = T.distributions.kl_divergence(T.distributions.Categorical(old_probs_batch),T.distributions.Categorical(new_probs)).mean()
            joint_loss = aux_loss + self.beta_clone * kl_loss

            self.policy_network_optim.zero_grad()
            joint_loss.backward()

            if self.max_grad_norm:
                clip_grad_norm_(self.policy_network.parameters(),self.max_grad_norm)
            self.policy_network_optim.step()
        
        # Optimize Lvalue wrt theta_V, on all data in buffer
        self.train_value(observations,returns,self.n_aux_batches)

    def _calculate_advantages(self, rewards: np.ndarray, values: np.ndarray, terminals: np.ndarray, last_values: np.ndarray):
        adv_arr = np.zeros(
            (self.n_workers, self.step_size+1), dtype=np.float32)
        next_val: float
        # TODO
        for i in range(self.n_workers):
            for t in reversed(range(self.step_size)):
                current_reward = rewards[i][t]
                current_val = values[i][t]
                if t == self.step_size-1:  # last step
                    next_val = last_values[i]
                else:
                    next_val = values[i][t+1]
                delta = current_reward + \
                    (self.gamma * next_val *
                     (1-int(terminals[i][t]))) - current_val
                adv_arr[i][t] = delta + (self.gamma*self.gae_lam *
                                         adv_arr[i][t+1] * (1-int(terminals[i][t])))
        adv_arr = adv_arr[:, :-1]
        advantages = T.tensor(
            adv_arr.flatten(), dtype=T.float32, device=self.device)

        return advantages

    def plot_summary(self,x:list,y:list,err:list,xlabel:str,ylabel:str,file_name:str):
        fig , ax = plt.subplots()
        ax.errorbar(x,y,yerr=err,linewidth=2.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.savefig(os.path.join("tmp",f"ppg-{file_name}"))
