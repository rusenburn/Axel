import os
import time
import gym
import numpy as np
import torch as T
import matplotlib.pyplot as plt
from collections import deque
from torch.nn.utils.clip_grad import clip_grad_norm_
from axel.common.networks import CnnActorCriticNetwork, SmallActorCriticNetwork,ImpalaActorCritic
from axel.common.running_normalizer import RunningNormalizer
from ..ppo.memory_buffer import MemoryBuffer
from axel.common.utils import calculate_explained_variance, seconds_to_HMS


class Pop3d:
    def __init__(self, vec_env: gym.vector.VectorEnv,
                 total_steps=int(1e5),
                 step_size=128,
                 n_batches=4,
                 n_epochs=4,
                 gamma=0.99,
                 gae_lam=0.95,
                 lr=2.5e-4,
                 entropy_coef=0.0,
                 critic_coef=0.5,
                 beta=10,
                 max_grad_norm=0.5,
                 normalize_rewards=False,
                 normalize_adv=False,
                 max_reward_norm=3,
                 decay_lr=True,
                 residual = True
                 ) -> None:

        # Environment info
        self.vec_env = vec_env
        self.n_workers = vec_env.num_envs
        self.observation_shape = vec_env.single_observation_space.shape
        self.n_game_actions = vec_env.single_action_space.n

        assert (self.n_workers*step_size) % n_batches == 0

        # Data Collection
        self.total_steps = int(total_steps)
        self.step_size = int(step_size)

        # Training Hyperparameters
        self.n_batches = int(n_batches)
        self.n_epochs = int(n_epochs)
        self.gamma = gamma
        self.gae_lam = gae_lam
        self.lr = lr
        self.beta = beta
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.decay_lr = decay_lr

        # Improvements
        self.max_grad_norm = max_grad_norm
        self.normalize_adv = normalize_adv
        self.normalize_rewards = normalize_rewards
        self.max_reward_norm = max_reward_norm
        self.rewards_normalizer = RunningNormalizer(maxlen=100,gamma=gamma)

        self.memory_buffer = MemoryBuffer(
            self.vec_env.single_observation_space.shape, self.n_workers, step_size)

        if len(self.observation_shape) > 2:
            if residual:
                self.network = ImpalaActorCritic(self.vec_env.single_observation_space.shape, self.n_game_actions)
            else:
                self.network = CnnActorCriticNetwork(
                    self.vec_env.single_observation_space.shape, self.n_game_actions)

        else:
            self.network = SmallActorCriticNetwork(
                self.n_game_actions, self.observation_shape)

        self.network_optim = T.optim.Adam(
            self.network.parameters(), lr=lr, eps=1e-5)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.network.to(self.device)

        # INFO
        self.rewards_std = deque(maxlen=100)
        self.scaler = 1

    def run(self) -> None:
        t_start = time.perf_counter()

        z : tuple[np.ndarray,dict] = self.vec_env.reset()
        observations, infos = z
        total_scores = np.zeros((self.n_workers,), dtype=np.float32)
        s = deque([], 100)

        total_iterations = self.total_steps // self.n_workers

        if self.decay_lr:
            def lr_fn(x): return (total_iterations-x) / total_iterations
        else:
            def lr_fn(x): return 1

        network_scheduler = T.optim.lr_scheduler.LambdaLR(
            self.network_optim, lr_fn)

        summary_time_step = []
        summary_duration = []
        summary_score = []
        summary_std = []
        value_losses = deque(maxlen=100)
        policy_losses = deque(maxlen=100)
        entropies = deque(maxlen=100)
        total_losses = deque(maxlen=100)
        explained_variances = deque(maxlen=100)
        for iteration in range(1, total_iterations+1):
            self.network.eval()
            actions, log_probs, values = self.choose_actions(observations)
            new_observations, rewards, dones, truncs, infos = self.vec_env.step(
                actions=actions)
            total_scores += rewards
            for i, (d, tr) in enumerate(list(zip(dones, truncs))):
                if d or tr:
                    s.append(total_scores[i])
                    rewards[i] = 0
                    total_scores[i] = 0
            self.memory_buffer.save(
                observations, actions, log_probs, values, rewards, dones)
            observations = new_observations
            if iteration % self.step_size == 0:
                last_values_tensor: T.Tensor
                with T.no_grad():
                    last_values_tensor = self.network.evaluate(
                        T.tensor(observations, dtype=T.float32, device=self.device))
                    last_values_ar: np.ndarray = last_values_tensor.squeeze().cpu().numpy()

                v_loss, p_loss, ent, total_loss , explained_variance = self.train_network(
                    last_values_ar)

                value_losses.append(v_loss)
                policy_losses.append(p_loss)
                entropies.append(ent)
                total_losses.append(total_loss)
                explained_variances.append(explained_variance)

                self.memory_buffer.reset()
                self.network.save_model(os.path.join("tmp", "network.pt"))
                T.save(self.network_optim, os.path.join(
                    "tmp", "network_optim.pt"))

            if (iteration * self.n_workers) % 10_000 < self.n_workers:
                total_duration = time.perf_counter()-t_start
                steps_done = self.n_workers*iteration

                score_mean = np.mean(s) if len(s) else 0
                score_std = np.std(s) if len(s) else 0
                score_err = 1.95 * score_std/(len(s)**0.5) if len(s) else 0
                fps = steps_done//total_duration

                v_loss: float = np.mean(value_losses)
                p_loss: float = np.mean(policy_losses)
                ent: float = np.mean(entropies)
                total_loss: float = np.mean(total_losses)
                hours,minutes,seconds =  seconds_to_HMS(total_duration)
                print(f"**************************************************")
                print(
                    f"Iteration:      {iteration} of {int(total_iterations)}")
                print(
                    f"learning rate:  {network_scheduler.get_last_lr()[0]:0.3e}")
                print(
                    f"Total Steps:    {steps_done} of {int(self.total_steps)}")
                print(f"Total duration: {hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}")
                print(f"Average Score:  {score_mean:0.2f} ± {score_err:0.2f}")
                print(f"Average FPS:    {fps}")
                print(f"Total Loss:     {total_loss:0.3f}")
                print(f"Value Loss:     {v_loss:0.3f}")
                print(f"Policy Loss:    {p_loss:0.3f}")
                print(f"Entropy:        {ent:0.3f}")
                print(f"Reward Scaler:  {self.scaler:0.3f}")
                value_losses.clear()
                entropies.clear()
                policy_losses.clear()
                total_losses.clear()

            if (iteration * self.n_workers) % 1_000 < self.n_workers:
                total_duration = time.perf_counter()-t_start
                steps_done = self.n_workers*iteration
                score_mean = np.mean(s) if len(s) else 0
                score_std = np.std(s) if len(s) else 0
                score_err = 1.95 * score_std/(len(s)**0.5) if len(s) else 0
                summary_time_step.append(steps_done)
                summary_duration.append(int(total_duration))
                summary_std.append(score_std)
                summary_score.append(score_mean)
            network_scheduler.step()

        errs = [1.95*std/len(summary_std)**0.5 for std in summary_std]
        self.plot_summary(summary_time_step, summary_score, errs,
                          "Time Step", "Score Last 100 games", "Score-Step")
        self.plot_summary(summary_duration, summary_score, errs,
                          "duration in seconds", "Score Last 100 games", "Score-duration")

    def choose_actions(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        observation_tensor = T.tensor(
            observations, dtype=T.float32, device=self.device)
        probs: T.Tensor
        values: T.Tensor
        with T.no_grad():
            probs, values = self.network.act_and_eval(observation_tensor)
        action_sample = T.distributions.Categorical(probs)
        actions = action_sample.sample()
        log_probs: T.Tensor = action_sample.log_prob(actions)
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.squeeze(-1).cpu().numpy()

    def train_network(self, last_values: np.ndarray):
        # self.network.train
        self.network.train()
        sample_size = self.n_workers*self.step_size
        batch_size = (sample_size) // self.n_batches

        observation_sample, action_sample, log_prob_samples,\
            value_samples, reward_samples, terminal_samples = self.memory_buffer.sample(
                step_based=True)

        if self.normalize_rewards:
            reward_samples ,self.scaler = self.rewards_normalizer.normalize_rewards(reward_samples,terminal_samples,self.max_reward_norm)

        all_advantages_ar, all_returns_ar = self._calc_adv(
            reward_samples, value_samples, terminal_samples, last_values)

        all_observation_ar = observation_sample.reshape(
            (sample_size, *self.observation_shape))
        all_actions_arr = action_sample.reshape(
            (sample_size,))
        all_log_probs_arr = log_prob_samples.reshape(
            (sample_size,))
        all_values_ar = value_samples.reshape(
            (sample_size,))

        all_observation_tensor = T.tensor(
            all_observation_ar, dtype=T.float32, device=self.device)
        all_actions_tensor = T.tensor(
            all_actions_arr, dtype=T.int32, device=self.device)
        all_log_probs_tensor = T.tensor(
            all_log_probs_arr, dtype=T.float32, device=self.device)
        all_values_tensor = T.tensor(
            all_values_ar, dtype=T.float32, device=self.device)
        all_advantages_tensor: T.Tensor = T.tensor(
            all_advantages_ar.flatten(), dtype=T.float32, device=self.device)
        all_returns_tensor: T.Tensor = T.tensor(
            all_returns_ar.flatten(), dtype=T.float32, device=self.device)
        if self.normalize_adv:
            all_advantages_tensor = (
                all_advantages_tensor - all_advantages_tensor.mean()) / (all_advantages_tensor.std()+1e-8)

        entropy_info = T.zeros(
            (self.n_epochs, self.n_batches), dtype=T.float32, device=self.device)
        value_losses_info = T.zeros(
            (self.n_epochs, self.n_batches), dtype=T.float32, device=self.device)
        policy_losses_info = T.zeros(
            (self.n_epochs, self.n_batches), dtype=T.float32, device=self.device)
        total_losses_info = T.zeros(
            (self.n_epochs, self.n_batches), dtype=T.float32, device=self.device)
        
        explained_variance = calculate_explained_variance(all_values_tensor,all_returns_tensor)
        for epoch in range(self.n_epochs):
            batch_starts = np.arange(
                0, self.n_workers*self.step_size, batch_size)
            indices = np.arange(self.n_workers*self.step_size)
            np.random.shuffle(indices)

            batches = [indices[i:i+batch_size] for i in batch_starts]
            for i, batch in enumerate(batches):
                observation_batch = all_observation_tensor[batch]
                old_log_prob_batch: T.Tensor = all_log_probs_tensor[batch]
                actions_batch = all_actions_tensor[batch]
                advantages_batch = all_advantages_tensor[batch]
                returns_batch = all_returns_tensor[batch]

                probs, predicted_values  = self.network.act_and_eval(observation_batch)
                predicted_values = predicted_values.squeeze()

                # get entropy
                probs_dist = T.distributions.Categorical(probs)
                entropy: T.Tensor = probs_dist.entropy().mean()

                # get policy loss
                predicted_log_probs: T.Tensor = probs_dist.log_prob(
                    actions_batch)
                predicted_probs = predicted_log_probs.exp()
                old_probs = old_log_prob_batch.exp()

                dpp = ((old_probs-predicted_probs)**2).mean()

                prob_ratio: T.Tensor = predicted_probs/old_probs

                weighted_prob_ratio: T.Tensor = (
                    prob_ratio * advantages_batch).mean()

                policy_loss = self.beta * dpp - weighted_prob_ratio

                critic_loss = 0.5 * \
                    ((returns_batch - predicted_values)**2).mean()
                total_loss: T.Tensor = policy_loss + self.critic_coef * \
                    critic_loss + self.entropy_coef * entropy
                self.network.zero_grad()
                total_loss.backward()

                if self.max_grad_norm:
                    clip_grad_norm_(self.network.parameters(),
                                    self.max_grad_norm)
                self.network_optim.step()

                with T.no_grad():
                    value_losses_info[epoch, i] = critic_loss
                    policy_losses_info[epoch, i] = policy_loss
                    entropy_info[epoch, i] = entropy
                    total_losses_info[epoch, i] = total_loss
        mean_value_loss = value_losses_info.flatten().mean()
        mean_policy_loss = policy_losses_info.flatten().mean()
        mean_entropy = entropy_info.flatten().mean()
        mean_total_loss = total_losses_info.flatten().mean()

        return mean_value_loss.cpu().item(), mean_policy_loss.cpu().item(), mean_entropy.cpu().item(), mean_total_loss.cpu().item(),explained_variance

    def _calc_adv(self, rewards: np.ndarray, values: np.ndarray, terminals: np.ndarray, last_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sample_size = len(values)
        advs = np.zeros_like(values)
        returns = np.zeros_like(values)

        next_val = last_values.copy()
        next_adv = np.zeros_like(last_values)
        gamma = self.gamma
        gae_lam = self.gae_lam

        for t in reversed(range(sample_size)):
            current_terminals: np.ndarray = terminals[t]
            current_rewards: np.ndarray = rewards[t]
            current_values: np.ndarray = values[t]

            next_val[current_terminals] = 0
            next_adv[current_terminals] = 0

            delta = current_rewards + gamma * next_val - current_values

            current_adv = delta + gamma * gae_lam * next_adv
            next_val = current_values.copy()
            next_adv = current_adv.copy()

            advs[t] = current_adv
            returns[t] = current_adv + current_values

        return advs, returns

    def _normalize_rewards(self, rewards: np.ndarray, max_norm: float) -> np.ndarray:
        # normalize rewards by dividing rewards by average moving standard deviation and clip any  value not in range [-max_norm,max_norm]
        flat_rewards = rewards.flatten()
        current_rewards_std = flat_rewards.std()
        self.rewards_std.append(current_rewards_std)
        rewards = np.clip(
            rewards/(np.mean(self.rewards_std)+1e-8), -max_norm, max_norm)
        return rewards

    def plot_summary(self, x: list, y: list, err: list, xlabel: str, ylabel: str, file_name: str):
        fig, ax = plt.subplots()
        ax.errorbar(x, y, yerr=err, linewidth=2.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.savefig(os.path.join("tmp", "pop3d-"+file_name))
