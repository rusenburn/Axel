import os
import time
import gym
import tqdm
import torch as T
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from torch.nn.utils.clip_grad import clip_grad_norm_
from ..common.networks import ActorCriticNetwork, CriticNetwork, SmallActorCriticNetwork, SmallCriticNetwork
from .ppg_memory_buffer import PpgMemoryBuffer


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
                 lr=2.5e-4,
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 normalize_adv=True,
                 normalize_rewards=True,
                 max_reward_norm=3,
                 decay_lr=True
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
        self.entropy_coef = entropy_coef
        self.decay_lr = decay_lr 
        # self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.normalize_adv = normalize_adv
        self.normalize_rewards = normalize_rewards
        self.max_reward_norm = max_reward_norm
        # networks
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        if len(self.observation_shape)>2:
            self.policy_network = ActorCriticNetwork(
            self.observation_shape, self.n_game_action)
            self.value_network = CriticNetwork(self.observation_shape)
        else:
            self.policy_network = SmallActorCriticNetwork(self.n_game_action,self.observation_shape)
            self.value_network = SmallCriticNetwork(self.observation_shape)
        self.policy_network_optim = T.optim.Adam(
            self.policy_network.parameters(), lr=lr)
        self.value_network_optim = T.optim.Adam(
            self.value_network.parameters(), lr=lr)

        self.policy_network.to(device=self.device)
        self.value_network.to(device=self.device)

        self.memory_buffer = PpgMemoryBuffer(
            self.observation_shape, self.n_workers, self.step_size,n_pi)
        
        self.rewards_mean = deque(maxlen=100)
        self.rewards_std = deque(maxlen=100)

    def run(self):
        # each phase consist of n_pi iterations which consist of n_workers * step_size steps
        # steps_per_phase = n_pi * n_workers * step_size
        steps_per_phase = self.n_pi * self.n_workers * self.step_size
        print(f"Steps per phase is {steps_per_phase}")
        n_phases = int(self.total_steps) // steps_per_phase
        if self.decay_lr:
            def lr_fn(x): return (n_phases-x) / (n_phases)
        else:
            def lr_fn(x): return 1
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
            value_losses_info = []
            policy_losses_info = []
            entropies = []

            for iteration in tqdm.tqdm(range(1,self.n_pi+1),f"Phase {phase} iterations:"):
                last_values_ar: np.ndarray
                for t in range(1,self.step_size+1):
                    actions, log_probs, values = self.choose_actions(
                        observations)
                    new_observations, rewards, dones, truncs, infos = self.vec_env.step(
                        actions)
                    total_score += rewards
                    for i, (d,tr) in enumerate(list(zip(dones,truncs))):
                        if d or tr:
                            s.append(total_score[i])
                            # I had to make the reward 0 when the env is done or trunc
                            rewards[i] = 0
                            total_score[i] = 0
                    self.memory_buffer.save_ppo_data(
                        observations, actions, log_probs, values, rewards, dones)
                    observations = new_observations
                    if t == self.step_size:  # last step
                        last_values_t: T.Tensor
                        with T.no_grad():
                            last_values_t = self.value_network(
                                T.tensor(observations, dtype=T.float32, device=self.device))
                            last_values_ar = last_values_t.cpu().numpy()

                observation_sample, action_sample, log_probs_sample, value_sample, reward_sample, dones_sample = self.memory_buffer.sample_ppo_data()

                if self.normalize_rewards:
                    reward_sample = self._normalize_rewards(reward_sample,self.max_reward_norm)

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
                    policy_loss,entropy = self.train_policy(observations_tensor, actions_tensor,
                                  log_probs_tensor, advantages_tensor)
                    policy_losses_info.append(policy_loss)
                    entropies.append(entropy)
                
                for ep in range(self.n_v_epochs):
                    value_loss= self.train_value(observations_tensor,
                                 value_tensor + advantages_tensor,self.n_batches)
                    value_losses_info.append(value_loss)
                self.memory_buffer.append_aux_data(observations_tensor.cpu().numpy(),(value_tensor+advantages_tensor).cpu().numpy())
                self.memory_buffer.reset_ppo_data()
            

            # prepare to train aux
            # get all data collected in the current phase
            all_obs_ar , returns_ar = self.memory_buffer.sample_aux_data()
            n_examples = len(all_obs_ar)
            returns_tensor = T.tensor(returns_ar,dtype=T.float32,device=self.device)
            all_old_probs_dist = T.zeros((n_examples,self.n_game_action),dtype=T.float32,device=self.device)
            batch_id = 0
            batch_size = 1024*4
            # evaluate all old observation probabilities distributions over actions
            # use batches to avoid full memory errors
            while batch_id < len(all_obs_ar)/batch_size:
                start = batch_id*batch_size
                end = (batch_id+1) * batch_size
                end = end if end < len(all_obs_ar) else len(all_obs_ar)
                obs_batch = all_obs_ar[start:end]
                obs_batch_tensor = T.tensor(obs_batch,dtype=T.float32,device=self.device)
                with T.no_grad():
                    probs:T.Tensor = self.policy_network(obs_batch_tensor)[0]
                all_old_probs_dist[start:end].copy_(probs.detach())
                batch_id+=1
            assert returns_tensor.device == self.device

            aux_value_losses_info = []
            aux_kl_divergence_info = []
            aux_joint_losses_info = []
            for ep in range(self.n_aux_epochs):
                aux_value_loss ,aux_kl_divergence,aux_joint_loss = self.train_aux(all_obs_ar,all_old_probs_dist, returns_tensor)
                aux_value_losses_info.append(aux_value_loss)
                aux_kl_divergence_info.append(aux_kl_divergence)
                aux_joint_losses_info.append(aux_joint_loss)

            self.memory_buffer.reset_aux_data()
            policy_network_scheduler.step()
            value_network_scheduler.step()

            if T.cuda.is_available():
                T.cuda.empty_cache()
            if phase % 1 == 0:
                self.policy_network.save_model(os.path.join("tmp","ppg_policy.pt"))
                self.value_network.save_model(os.path.join("tmp","ppg_value.pt"))
            score_mean = np.mean(s) if len(s) else 0
            score_std = np.std(s) if len(s) else 0
            score_err = (1.95 * score_std / (len(s)**0.5)) if len(s) else 0
            steps_done = phase * steps_per_phase
            total_duration = time.perf_counter()-t_start
            value_loss = np.mean(value_losses_info)
            policy_loss = np.mean(policy_losses_info)
            entropy = np.mean(entropies)

            aux_value_loss = np.mean(aux_value_losses_info)
            aux_kl_divergence = np.mean(aux_kl_divergence_info)
            aux_joint_losses_info = np.mean(aux_joint_loss)
            fps = steps_done // total_duration
            if phase % 1 == 0:
                print(f"*************************************")
                print(f"Phase:          {phase} of {n_phases}")
                print(f"learning_rate:  {policy_network_scheduler.get_last_lr()[0]:0.2e}")
                print(f"Total Steps:    {phase*steps_per_phase} of {self.total_steps}")
                print(f"Total duration: {total_duration:0.2f} seconds")
                print(f"Average Score:  {score_mean:0.2f} ± {score_err:0.2f}")
                print(f"Average FPS:    {int(fps)}")
                print(f"Value Loss:     {value_loss:0.3f}")
                print(f"Policy Loss:    {policy_loss:0.2e}")
                print(f"Entropy:        {entropy:0.3f}")
                print(f"Aux Value Loss: {aux_value_loss:0.3f}")
                print(f"Aux KL-D:       {aux_kl_divergence:0.3f}")
                print(f"Aux Joint Loss: {aux_joint_loss:0.3f}")


            summary_duration.append(total_duration)
            summary_time_step.append(phase*steps_per_phase)
            summary_score.append( score_mean)
            summary_err.append(score_err)
        

        self.plot_summary(summary_time_step,summary_score,summary_err,"Steps","Score","Step-Score.png")
        self.plot_summary(summary_duration,summary_score,summary_err,"Duration in seconds","Score","Duration-Score.png")

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
    
        batch_starts = np.arange(0, self.n_workers*self.step_size,batch_size)
        indices = np.arange(self.n_workers*self.step_size)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_starts]
        if self.normalize_adv:
                advantages = (advantages - advantages.mean()) / (advantages.std()+1e-8)
        entropy_losses = T.zeros((self.n_batches,),dtype=T.float32,device=self.device)
        policy_losses = T.zeros((self.n_batches,),dtype=T.float32,device=self.device)
        for i,batch in enumerate(batches):
            observation_batch = observation[batch]
            old_log_probs_batch = log_probs[batch]
            actions_batch = actions[batch]
            advantages_batch = advantages[batch].clone()
            # if self.normalize_adv:
            #     advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std()+1e-8)
            
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
            loss = lclip - self.entropy_coef*entropy

            self.policy_network.zero_grad()
            loss.backward()

            if self.max_grad_norm:
                clip_grad_norm_(self.policy_network.parameters(),max_norm=self.max_grad_norm)

            self.policy_network_optim.step()
            policy_losses[i] = lclip
            entropy_losses[i] = entropy

        with T.no_grad():
            policy_loss = policy_losses.mean().cpu().item()
            entropy_loss = entropy_losses.mean().cpu().item()
        return policy_loss,entropy_loss

    def train_value(self,observations:T.Tensor|np.ndarray,returns:T.Tensor,n_batches:int):
        size = observations.shape[0]
        batch_size = size // n_batches
        batch_starts = np.arange(0, size,batch_size)
        indices = np.arange(size)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_starts]
        losses = T.zeros((n_batches,),dtype=T.float32,device=self.device)
        for i,batch in enumerate(batches):
            observations_batch:T.Tensor|np.ndarray = observations[batch]
            if not isinstance(observations_batch,T.Tensor):
                observations_batch = T.tensor(observations_batch,dtype=T.float32,device=self.device)
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
            losses[i]=value_loss
        
        with T.no_grad():
            l = losses.mean().cpu().item()
        return l
        
        

    def train_aux(self,observations:np.ndarray,old_probs:T.Tensor,returns:T.Tensor):
        size = observations.shape[0]
        n_batches = self.n_aux_batches
        batch_size = size // n_batches
        batch_starts = np.arange(0, size,batch_size)
        indices = np.arange(size)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_starts]

        aux_value_losses = T.zeros((n_batches,),dtype=T.float32,device=self.device)
        aux_kl_divergence = T.zeros((n_batches,),dtype=T.float32,device=self.device)
        aux_joint_losses = T.zeros((n_batches,),dtype=T.float32,device=self.device)
        # Optimize Ljoint wrt theta_pi, on all data in buffer    
        for i,batch in enumerate(batches):
            old_probs_batch = old_probs[batch]
            observations_batch_ar = observations[batch]
            returns_batch = returns[batch]
            observations_batch = T.tensor(observations_batch_ar,dtype=T.float32,device=self.device)
            predicted_policy_values:T.Tensor
            new_probs: T.Tensor
            new_probs,predicted_policy_values = self.policy_network(observations_batch)
            predicted_policy_values = predicted_policy_values.squeeze(-1)

            aux_loss = 0.5 * ((returns_batch - predicted_policy_values)**2).mean()
            kl_loss = T.distributions.kl_divergence(T.distributions.Categorical(old_probs_batch),T.distributions.Categorical(new_probs)).mean()
            joint_loss = aux_loss + self.beta_clone * kl_loss
            self.policy_network_optim.zero_grad()
            joint_loss.backward()
            if self.max_grad_norm:
                clip_grad_norm_(self.policy_network.parameters(),self.max_grad_norm)
            self.policy_network_optim.step()

            with T.no_grad():
                aux_value_losses[i] = aux_loss
                aux_kl_divergence[i] = kl_loss
                aux_joint_losses[i] = joint_loss
                
        # Train Critic network on all data in buffer
        # Optimize Lvalue wrt theta_V, on all data in buffer
        self.train_value(observations,returns,self.n_aux_batches)

        # logging info
        mean_value_loss = aux_value_losses.mean().cpu().item()
        mean_kl_divergence = aux_kl_divergence.mean().cpu().item()
        mean_joint_loss = aux_joint_losses.mean().cpu().item()
        return mean_value_loss,mean_kl_divergence,mean_joint_loss

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


    def _normalize_rewards(self,rewards:np.ndarray,max_norm=10)->np.ndarray:
        # normalize rewards by dividing rewards by their standard deviation and clip any  value not in range (-max_norm,max_norm)
        flat_rewards = rewards.flatten()
        current_rewards_std = flat_rewards.std()
        current_rewards_mean = flat_rewards.mean()
        self.rewards_std.append(current_rewards_std)
        self.rewards_mean.append(current_rewards_mean)

        n_size = self.n_workers*self.step_size
        all_sums:float = 0
        all_squared_sums:float = 0
        all_counts:float = 0
        for rew_mean,rew_std in list(zip(self.rewards_mean,self.rewards_std)):
            sum_x:float = rew_mean * n_size
            sum_x_squared:float = (rew_std**2) * (n_size-1) + (sum_x**2) / n_size
            all_counts+=n_size
            all_sums += sum_x
            all_squared_sums+=sum_x_squared
        
        combined_mean = all_sums/all_counts
        combined_std = ((all_squared_sums-(all_sums**2)/all_counts)/(all_counts-1))**0.5
        combined_err = combined_std / (all_counts**0.5 + 1e-8)

        
        
        rewards_1 = np.clip((rewards - combined_mean-combined_err)/(combined_std+1e-8),-max_norm,max_norm)
        rewards_2 = np.clip((rewards - combined_mean+combined_err)/(combined_std+1e-8),-max_norm,max_norm)
        rewards = rewards_1*0.5 + rewards_2*0.5

        # rewards = np.clip(rewards/(max(1,flat_rewards.std())),-max_norm,max_norm)
        # rewards = np.clip(rewards/(flat_rewards.std()+1e-8),-max_norm,max_norm)
        # rewards = np.clip(rewards/(np.mean(self.rewards_std)+1e-8),-max_norm,max_norm)
        return rewards
    def plot_summary(self,x:list,y:list,err:list,xlabel:str,ylabel:str,file_name:str):
        fig , ax = plt.subplots()
        ax.errorbar(x,y,yerr=err,linewidth=2.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.savefig(os.path.join("tmp",f"ppg-{file_name}"))
