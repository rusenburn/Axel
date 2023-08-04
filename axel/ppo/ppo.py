import time
import gym.vector
import os
import numpy as np
import torch as T
import matplotlib.pyplot as plt
from collections import deque
from torch.optim import Adam
from torch.distributions import Categorical
from torch.nn.utils.clip_grad import clip_grad_norm_

from axel.common.running_normalizer import RunningNormalizer
from .memory_buffer import MemoryBuffer
from axel.common.utils import calculate_explained_variance, seconds_to_HMS
from axel.common.networks import ActorNetwork, CriticNetwork, ImpalaActorCritic, ImpalaCritic, NetworkBase, CnnActorNetwork, CnnCriticNetwork, SmallActorNetwork, SmallCriticNetwork


class Ppo:
    def __init__(self,
                 vec_env: gym.vector.VectorEnv,
                 total_steps=1_000_000,
                 step_size=128,
                 n_batches=4,
                 n_epochs=4,
                 gamma=0.99,
                 gae_lam=0.95,
                 policy_clip=0.2,
                 lr=2.5e-4,
                 entropy_coef=0.01,
                 critic_coef=0.5,
                 max_grad_norm=0.5,
                 normalize_adv=True,
                 normalize_rewards=True,
                 max_reward_norm = 3,
                 decay_lr=True,
                 residual=True) -> None:

        self.vec_env = vec_env
        self.n_workers = vec_env.num_envs
        self.n_game_actions = self.vec_env.single_action_space.n
        self.observation_shape = self.vec_env.single_observation_space.shape
        assert (self.n_workers*step_size) % n_batches == 0

        # Data Collections
        self.total_steps = int(total_steps)  # incase user gave float or str
        self.step_size = int(step_size)

        # Training Hyperparameters
        self.n_batches = int(n_batches)
        self.n_epochs = int(n_epochs)
        self.gamma = gamma
        self.gae_lam = gae_lam
        self.policy_clip = policy_clip
        self.lr = lr
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.decay_lr = decay_lr

        # Improvements
        self.max_grad_norm = max_grad_norm
        self.normalize_adv = normalize_adv
        self.normalize_rewards = normalize_rewards
        self.max_reward_norm = max_reward_norm
        self.reward_normalizer = RunningNormalizer(maxlen=100,gamma=gamma)

        self.memory_buffer = MemoryBuffer(
            self.vec_env.single_observation_space.shape, self.vec_env.num_envs, self.step_size)
        

        if len(self.observation_shape)>2:
            if residual:
                self.actor : ActorNetwork = ImpalaActorCritic(self.vec_env.single_observation_space.shape,self.n_game_actions)
                self.critic : CriticNetwork = ImpalaCritic(self.vec_env.single_observation_space.shape)
            else:
                self.actor : ActorNetwork = CnnActorNetwork(
                    self.vec_env.single_observation_space.shape, self.n_game_actions)
                self.critic : CriticNetwork = CnnCriticNetwork(
                    self.vec_env.single_observation_space.shape)
        else:
            self.actor = SmallActorNetwork(self.n_game_actions,self.observation_shape)
            self.critic = SmallCriticNetwork(self.observation_shape)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr, eps=1e-5)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr, eps=1e-5)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.actor.to(self.device)
        self.critic.to(self.device)

        # INFO
        self.rewards_std = deque(maxlen=50)
        self.rewards_mean = deque(maxlen=50)
        self.scaler = 1

    def run(self) -> None:
        t_start = time.perf_counter()
        observations: np.ndarray
        infos: dict
        observations, infos = self.vec_env.reset()
        total_scores = np.zeros((self.n_workers,), dtype=np.float32)
        s = deque([], 100)
        total_iterations = int(self.total_steps//self.n_workers)
        if self.decay_lr:
            def lr_fn(x: int): return (total_iterations-x) / total_iterations
        else:
            def lr_fn(x: int): return 1.0
        actor_scheduler = T.optim.lr_scheduler.LambdaLR(
            self.actor_optim, lr_fn)
        critic_shceduler = T.optim.lr_scheduler.LambdaLR(
            self.critic_optim, lr_fn)

        summary_time_step = []
        summary_duration = []
        summary_score = []
        summary_err = []

        value_losses = deque(maxlen=100)
        policy_losses = deque(maxlen=100)
        entropies = deque(maxlen=100)
        total_losses = deque(maxlen=100)
        explained_variances = deque(maxlen=100)
        for iteration in range(1, int(self.total_steps//self.n_workers)+1):
            actions, log_probs, values = self.choose_actions(observations)
            new_observations, rewards, dones, truncs, infos = self.vec_env.step(
                actions=actions)
            total_scores += rewards
            for i, (d,tr) in enumerate(list(zip(dones,truncs))):
                if d or tr:
                    s.append(total_scores[i])
                    rewards[i] = 0
                    total_scores[i] = 0
            self.memory_buffer.save(
                observations, actions, log_probs, values, rewards, dones)
            observations = new_observations
            if iteration % self.step_size == 0:
                with T.no_grad():
                    last_values_t: T.Tensor = self.critic(
                        T.tensor(observations.copy(), dtype=T.float32, device=self.device))
                    last_values_ar: np.ndarray = last_values_t.squeeze().cpu().numpy()
                p_loss ,v_loss,ent , total_loss,explained_variance = self.train_network(last_values_ar)
                value_losses.append(v_loss)
                policy_losses.append(p_loss)
                entropies.append(ent)
                total_losses.append(total_loss)
                explained_variances.append(explained_variance)
                self.memory_buffer.reset()
                self.actor.save_model(os.path.join("tmp", "actor.pt"))
                self.critic.save_model(os.path.join("tmp", "critic.pt"))
                T.save(self.actor_optim, os.path.join("tmp", "actor_optim.pt"))
                T.save(self.critic_optim, os.path.join(
                    "tmp", "critic_optim.pt"))

            steps_done = self.n_workers * iteration
            if iteration*self.n_workers % 2_000 < self.n_workers:
                total_duration = time.perf_counter()-t_start
                fps = steps_done // total_duration
                score_mean = np.mean(s) if len(s) > 0 else 0
                score_std = np.std(s) if len(s) > 0 else 0
                score_err = 1.95 * score_std / \
                    (len(s)**0.5) if len(s) > 0 else 0
                total_loss = np.mean(total_losses)
                v_loss = np.mean(value_losses)
                p_loss = np.mean(policy_losses)
                ent = np.mean(entropies)
                explained_variance = np.mean(explained_variances)
                hours,minutes,seconds = seconds_to_HMS(total_duration)
                print(f"**************************************************")
                print(f"Iteration:      {iteration} of {total_iterations}")
                print(f"Learning rate:  {actor_scheduler.get_last_lr()[0]:0.3e}")
                print(f"FPS:            {fps}")
                print(
                    f"Total Steps:    {steps_done} of {int(self.total_steps)}")
                print(f"Total duration: {hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}")
                print(f"Average Score:  {score_mean:0.2f} ± {score_err:0.2f}")
                print(f"Total Loss:     {total_loss:0.3f}")
                print(f"Value Loss:     {v_loss:0.3f}")
                print(f"Policy Loss:    {p_loss:0.3f}")
                print(f"Entropy:        {ent:0.3f}")
                print(f"E-Var-ratio:    {explained_variance:0.3f}")
                print(f"Reward Scaler:  {self.scaler:0.3f}")

                summary_duration.append(total_duration)
                summary_time_step.append(steps_done)
                summary_score.append(score_mean)
                summary_err.append(score_err)

                value_losses.clear()
                policy_losses.clear()
                entropies.clear()
                total_losses.clear()
                explained_variances.clear()

            actor_scheduler.step()
            critic_shceduler.step()

        self.plot_summary(summary_time_step, summary_score,
                          summary_err, "Steps", "Score", "Step-Score.png")
        self.plot_summary(summary_duration, summary_score,
                          summary_err, "Steps", "Score", "Duration-Score.png")

    def choose_actions(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state: T.Tensor = T.tensor(obs, dtype=T.float32, device=self.device)
        with T.no_grad():
            probs: T.Tensor = self.actor.act(state)
            values: T.Tensor = self.critic.evaluate(state)

        action_sample = Categorical(probs)
        actions = action_sample.sample()
        log_probs: T.Tensor = action_sample.log_prob(actions)
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.squeeze(-1).cpu().numpy()

    def train_network(self, last_values: np.ndarray):
        # self.train
        sample_size = self.n_workers*self.step_size
        batch_size = (sample_size) // self.n_batches
        state_samples, action_samples, log_prob_samples, \
                value_samples, reward_samples, terminal_samples = self.memory_buffer.sample(step_based=True)
        
        if self.normalize_rewards:
            reward_samples , self.scaler = self.reward_normalizer.normalize_rewards(reward_samples,terminal_samples,self.max_reward_norm)

        advantages_arr , returns_arr = self._calc_adv(reward_samples,value_samples,terminal_samples,last_values)

        all_advantages_tensor = T.tensor(advantages_arr.flatten(),dtype=T.float32,device=self.device)
        all_returns_tensor = T.tensor(returns_arr.flatten(),dtype=T.float32,device=self.device)

        # normalize adv
        if self.normalize_adv:
                    all_advantages_tensor = (
                        all_advantages_tensor - all_advantages_tensor.mean())/(all_advantages_tensor.std()+1e-8)

        value_loss_info = T.zeros((self.n_epochs,self.n_batches),dtype=T.float32,device=self.device)
        policy_loss_info = T.zeros((self.n_epochs,self.n_batches,),dtype=T.float32,device=self.device)
        entropies_info = T.zeros((self.n_epochs,self.n_batches),dtype=T.float32,device=self.device)
        total_losses_info = T.zeros((self.n_epochs,self.n_batches),dtype=T.float32,device=self.device)

        all_states_arr = state_samples.reshape(
                (sample_size, *self.vec_env.single_observation_space.shape))
        all_actions_arr = action_samples.reshape(
            (sample_size,))
        all_log_probs_arr = log_prob_samples.reshape(
            (sample_size,))
        all_values_arr = value_samples.reshape(
            (sample_size,))
        
        all_states_tensor = T.tensor(all_states_arr, device=self.device)
        all_actions_tensor = T.tensor(all_actions_arr, device=self.device)
        all_log_probs_tensor = T.tensor(all_log_probs_arr, device=self.device)
        all_values_tensor = T.tensor(all_values_arr, device=self.device)

        exaplained_variance = calculate_explained_variance(all_values_tensor,all_returns_tensor)

        for epoch in range(self.n_epochs):
            batch_starts = np.arange(
                0, self.n_workers*self.step_size, batch_size)
            indices = np.arange(self.n_workers*self.step_size, dtype=np.int32)
            np.random.shuffle(indices)

            batches = [indices[i:i+batch_size] for i in batch_starts]

            for i,batch in enumerate(batches):
                states_tensor: T.Tensor = all_states_tensor[batch]
                old_log_probs_batch: T.Tensor = all_log_probs_tensor[batch]
                actions_batch: T.Tensor = all_actions_tensor[batch]
                advantages_batch = all_advantages_tensor[batch]
                returns_batch = all_returns_tensor[batch]

                # get predictions for current batch of states
                probs: T.Tensor = self.actor.act(states_tensor)
                critic_values: T.Tensor = self.critic.evaluate(states_tensor)
                critic_values = critic_values.squeeze()

                # get entropy
                dist = Categorical(probs)
                entropy: T.Tensor = dist.entropy().mean()

                # find actor loss
                new_log_probs: T.Tensor = dist.log_prob(actions_batch)
                prob_ratio = (new_log_probs-old_log_probs_batch).exp()
                weighted_probs = advantages_batch * prob_ratio
                clipped_prob_ratio = prob_ratio.clamp(
                    1-self.policy_clip, 1+self.policy_clip)
                weighted_clipped_probs = advantages_batch * clipped_prob_ratio
                actor_loss = -T.min(weighted_probs,
                                    weighted_clipped_probs).mean()

                # Find critic loss
                old_values_predictions = all_values_tensor[batch]
                clipped_values_predictions = old_values_predictions + \
                    T.clamp(critic_values-old_values_predictions, -
                            self.policy_clip, self.policy_clip)
                critic_loss = (returns_batch - critic_values)**2
                critic_loss_2 = (returns_batch - clipped_values_predictions)**2
                critic_loss = 0.5*T.max(critic_loss, critic_loss_2).mean()
                total_loss = actor_loss + self.critic_coef * critic_loss - self.entropy_coef * entropy

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                total_loss.backward()

                if self.max_grad_norm:
                    clip_grad_norm_(self.actor.parameters(),
                                    max_norm=self.max_grad_norm)
                    clip_grad_norm_(self.critic.parameters(),
                                    max_norm=self.max_grad_norm)
                self.actor_optim.step()
                self.critic_optim.step()
                with T.no_grad():
                    value_loss_info[epoch,i] = critic_loss
                    policy_loss_info[epoch,i] = actor_loss
                    entropies_info[epoch,i] = entropy
                    total_losses_info[epoch,i] = total_loss
            
        with T.no_grad():
            value_loss_mean = value_loss_info.flatten().mean()
            policy_loss_mean = policy_loss_info.flatten().mean()
            entropy_mean= entropies_info.flatten().mean()
            total_loss_mean = total_losses_info.flatten().mean()
        return policy_loss_mean.cpu().item(),value_loss_mean.cpu().item(),entropy_mean.cpu().item(),total_loss_mean.cpu().item(),exaplained_variance
            

    def _calc_adv(self,rewards:np.ndarray,values:np.ndarray,terminals:np.ndarray,last_values:np.ndarray)->tuple[np.ndarray,np.ndarray]:
        sample_size = len(values)
        advs = np.zeros_like(values)
        returns = np.zeros_like(values)

        next_val = last_values.copy()
        next_adv = np.zeros_like(last_values)
        gamma = self.gamma
        gae_lam = self.gae_lam

        for t in reversed(range(sample_size)):
            current_terminals :np.ndarray= terminals[t]
            current_rewards : np.ndarray = rewards[t]
            current_values : np.ndarray = values[t]

            next_val[current_terminals] = 0
            next_adv[current_terminals] = 0

            delta = current_rewards + gamma * next_val - current_values

            current_adv = delta + gamma * gae_lam * next_adv
            next_val = current_values.copy()
            next_adv = current_adv.copy()

            advs[t] = current_adv
            returns[t] = current_adv + current_values
        
        return advs,returns


    def plot_summary(self, x: list, y: list, err: list, xlabel: str, ylabel: str, file_name: str):
        fig, ax = plt.subplots()
        ax.errorbar(x, y, yerr=err, linewidth=2.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.savefig(os.path.join("tmp", f"ppo-{file_name}"))
    
    def _normalize_rewards(self,rewards:np.ndarray,max_norm)->np.ndarray:
        # normalize rewards by dividing rewards by average moving standard deviation and clip any  value not in range [-max_norm,max_norm]
        flat_rewards = rewards.flatten()
        current_rewards_std = flat_rewards.std()
        current_rewards_mean = flat_rewards.mean()
        self.rewards_std.append(current_rewards_std)
        self.rewards_mean.append(current_rewards_mean)
        rewards = np.clip(rewards /(np.mean(self.rewards_std)+1e-8),-max_norm,max_norm)
        return rewards