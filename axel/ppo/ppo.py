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
from .memory_buffer import MemoryBuffer
from ..common.networks import NetworkBase, ActorNetwork, CriticNetwork, SmallActorNetwork, SmallCriticNetwork


class Ppo:
    def __init__(self,
                 vec_env: gym.vector.VectorEnv,
                 total_steps=1e5,
                 step_size=5,
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
                 normalize_rewards=False,
                 max_reward_norm = 3,
                 decay_lr=True) -> None:

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
        self.max_grad_norm = 0.5
        self.normalize_rewards = normalize_rewards
        self.max_reward_norm = max_reward_norm

        self.memory_buffer = MemoryBuffer(
            self.vec_env.single_observation_space.shape, self.vec_env.num_envs, self.step_size)
        

        if len(self.observation_shape)>2:
            self.actor = ActorNetwork(
                self.vec_env.single_observation_space.shape, self.n_game_actions)
            self.critic = CriticNetwork(
                self.vec_env.single_observation_space.shape)
        else:
            self.actor = SmallActorNetwork(self.n_game_actions,self.observation_shape)
            self.critic = SmallCriticNetwork(self.observation_shape)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr, eps=1e-5)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr, eps=1e-5)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.actor.to(self.device)
        self.critic.to(self.device)

        self.rewards_std = deque(maxlen=100)
        self.rewards_mean = deque(maxlen=100)

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
        for iteration in range(1, int(self.total_steps//self.n_workers)+1):
            iteration_start = time.perf_counter()
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
                    last_values_ar: np.ndarray = last_values_t.cpu().numpy()
                p_loss ,v_loss,ent , total_loss = self.train_network(last_values_ar)
                value_losses.append(v_loss)
                policy_losses.append(p_loss)
                entropies.append(ent)
                total_losses.append(total_loss)
                self.memory_buffer.reset()
                self.actor.save_model(os.path.join("tmp", "actor.pt"))
                self.critic.save_model(os.path.join("tmp", "critic.pt"))
                T.save(self.actor_optim, os.path.join("tmp", "actor_optim.pt"))
                T.save(self.critic_optim, os.path.join(
                    "tmp", "critic_optim.pt"))

            if iteration*self.n_workers % 1_000 < self.n_workers:
                total_duration = time.perf_counter()-t_start
                steps_done = self.n_workers * iteration
                fps = steps_done // total_duration
                score_mean = np.mean(s) if len(s) > 0 else 0
                score_std = np.std(s) if len(s) > 0 else 0
                score_err = 1.95 * score_std / \
                    (len(s)**0.5) if len(s) > 0 else 0
                total_loss = np.mean(total_losses)
                v_loss = np.mean(value_losses)
                p_loss = np.mean(p_loss)
                ent = np.mean(entropies)
                print(f"**************************************************")
                print(f"Iteration:      {iteration} of {total_iterations}")
                print(f"Learning rate:  {actor_scheduler.get_last_lr()[0]:0.3e}")
                print(f"FPS:            {fps}")
                print(
                    f"Total Steps:    {steps_done} of {int(self.total_steps)}")
                print(f"Total duration: {total_duration:0.2f} seconds")
                print(f"Average Score:  {score_mean:0.2f} ± {score_err:0.2f}")
                print(f"Total Loss:     {total_loss:0.3f}")
                print(f"Value Loss:     {v_loss:0.3f}")
                print(f"Policy Loss:    {p_loss:0.3f}")
                print(f"Entropy:        {ent:0.3f}")

                summary_duration.append(total_duration)
                summary_time_step.append(steps_done)
                summary_score.append(score_mean)
                summary_err.append(score_err)

                value_losses.clear()
                policy_losses.clear()
                entropies.clear()
                total_losses.clear()


            actor_scheduler.step()
            critic_shceduler.step()

        self.plot_summary(summary_time_step, summary_score,
                          summary_err, "Steps", "Score", "Step-Score.png")
        self.plot_summary(summary_duration, summary_score,
                          summary_err, "Steps", "Score", "Duration-Score.png")

    def choose_actions(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state: T.Tensor = T.tensor(obs, dtype=T.float32, device=self.device)
        with T.no_grad():
            probs: T.Tensor = self.actor(state)
            values: T.Tensor = self.critic(state)

        action_sample = Categorical(probs)
        actions = action_sample.sample()
        log_probs: T.Tensor = action_sample.log_prob(actions)
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.squeeze(-1).cpu().numpy()

    def train_network(self, last_values: np.ndarray):
        # self.train
        batch_size = (self.step_size*self.n_workers) // self.n_batches
        state_batches, action_batches, log_prob_batches, \
                value_batches, reward_batches, terminal_batches = self.memory_buffer.sample()
        
        if self.normalize_rewards:
            reward_batches = self._normalize_rewards(reward_batches,self.max_reward_norm)

        advantages_tensor = self._calculate_advatages(
                reward_batches, value_batches, terminal_batches, last_values)
        
        # normalize adv
        if self.normalize_adv:
                    normalized_advantages_tensor = (
                        advantages_tensor - advantages_tensor.mean())/(advantages_tensor.std()+1e-8)

        value_loss_info = T.zeros((self.n_epochs,self.n_batches),dtype=T.float32,device=self.device)
        policy_loss_info = T.zeros((self.n_epochs,self.n_batches,),dtype=T.float32,device=self.device)
        entropies_info = T.zeros((self.n_epochs,self.n_batches),dtype=T.float32,device=self.device)
        total_losses_info = T.zeros((self.n_epochs,self.n_batches),dtype=T.float32,device=self.device)
        for epoch in range(self.n_epochs):
            batch_starts = np.arange(
                0, self.n_workers*self.step_size, batch_size)
            indices = np.arange(self.n_workers*self.step_size, dtype=np.int32)
            np.random.shuffle(indices)

            batches = [indices[i:i+batch_size] for i in batch_starts]

            states_arr = state_batches.reshape(
                (self.n_workers*self.step_size, *self.vec_env.single_observation_space.shape))
            actions_arr = action_batches.reshape(
                (self.n_workers*self.step_size,))
            log_probs_arr = log_prob_batches.reshape(
                (self.n_workers*self.step_size,))
            values_arr = value_batches.reshape(
                (self.n_workers*self.step_size,))
            values_tensor = T.tensor(values_arr, device=self.device)
            for i,batch in enumerate(batches):
                states_tensor: T.Tensor = T.tensor(
                    states_arr[batch], dtype=T.float32, device=self.device)
                old_log_probs_tensor: T.Tensor = T.tensor(
                    log_probs_arr[batch], dtype=T.float32, device=self.device)
                actions_tensor: T.Tensor = T.tensor(
                    actions_arr[batch], device=self.device)

                # Normalize advantages
                if self.normalize_adv:
                    normalized_advatages = normalized_advantages_tensor[batch].clone()
                else:
                    normalized_advatages = advantages_tensor[batch].clone()

                # get predictions for current batch of states
                probs: T.Tensor = self.actor(states_tensor)
                critic_values: T.Tensor = self.critic(states_tensor)
                critic_values = critic_values.squeeze()

                # get entropy
                dist = Categorical(probs)
                entropy: T.Tensor = dist.entropy().mean()

                # find actor loss
                new_log_probs: T.Tensor = dist.log_prob(actions_tensor)
                prob_ratio = (new_log_probs-old_log_probs_tensor).exp()
                weighted_probs = normalized_advatages * prob_ratio
                clipped_prob_ratio = prob_ratio.clamp(
                    1-self.policy_clip, 1+self.policy_clip)
                weighted_clipped_probs = normalized_advatages * clipped_prob_ratio
                actor_loss = -T.min(weighted_probs,
                                    weighted_clipped_probs).mean()

                # find critic loss ## NOTE We do not use the normalized advantages here
                returns = advantages_tensor[batch] + values_tensor[batch]
                old_values_predictions = values_tensor[batch]
                clipped_values_predictions = old_values_predictions + \
                    T.clamp(critic_values-old_values_predictions, -
                            self.policy_clip, self.policy_clip)
                critic_loss = (returns - critic_values)**2
                critic_loss_2 = (returns - clipped_values_predictions)**2
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
        return policy_loss_mean.cpu().item(),value_loss_mean.cpu().item(),entropy_mean.cpu().item(),total_loss_mean.cpu().item()
            

    def _calculate_advatages(self, rewards: np.ndarray, values: np.ndarray, terminals: np.ndarray, last_values: np.ndarray):
        adv_arr = np.zeros(
            (self.n_workers, self.step_size+1), dtype=np.float32)
        next_val: float
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
