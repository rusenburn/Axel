import os
import time
import gym
import numpy as np
import torch as T
import matplotlib.pyplot as plt
from collections import deque
from torch.nn.utils.clip_grad import clip_grad_norm_
from axel.common.networks import ActorCriticNetwork
from ..ppo.memory_buffer import MemoryBuffer


class Pop3d:
    def __init__(self, vec_env: gym.vector.VectorEnv,
                 total_steps=int(1e5), step_size=128, n_batches=4,
                 n_epochs=4, gamma=0.99, gae_lam=0.95, lr=2.5e-4, beta=5,
                 max_grad_norm=0.5,normalize_adv=False) -> None:

        self.vec_env = vec_env
        self.n_workers = vec_env.num_envs
        assert (self.n_workers*step_size) % n_batches == 0
        self.total_steps = total_steps
        self.step_size = step_size
        self.n_batches = n_batches
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lam = gae_lam
        self.lr = lr
        self.beta = beta
        self.max_grad_norm = max_grad_norm
        self.normalize_adv = normalize_adv

        self.memory_buffer = MemoryBuffer(
            self.vec_env.single_observation_space.shape, self.n_workers, step_size)
        
        self.observation_shape = vec_env.single_observation_space.shape
        self.n_game_actions = self.vec_env.single_action_space.n

        self.network = ActorCriticNetwork(
            self.vec_env.single_observation_space.shape, self.n_game_actions)
        self.network_optim = T.optim.Adam(
            self.network.parameters(), lr=lr, eps=1e-5)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.network.to(self.device)

    def run(self) -> None:
        t_start = time.perf_counter()

        observations: np.ndarray
        infos: dict

        observations, infos = self.vec_env.reset()
        total_scores = np.zeros((self.n_workers,), dtype=np.float32)
        s = deque([], 100)

        total_iterations = self.total_steps // self.n_workers

        def alpha_ratio(x): return (total_iterations-x) / total_iterations

        network_scheduler = T.optim.lr_scheduler.LambdaLR(
            self.network_optim, alpha_ratio)

        summary_time_step = []
        summary_duration = []
        summary_score = []
        summary_std = []
        for iteration in range(1, total_iterations+1):
            iteration_start = time.perf_counter()
            actions, log_probs, values = self.choose_actions(observations)
            new_observations, rewards, dones, truncs, infos = self.vec_env.step(
                actions=actions)
            self.memory_buffer.save(
                observations, actions, log_probs, values, rewards, dones)
            observations = new_observations
            total_scores += rewards

            for i, d in enumerate(dones):
                if d:
                    s.append(total_scores[i])
                    total_scores[i] = 0

            if iteration % self.step_size == 0:
                last_values_tensor: T.Tensor
                with T.no_grad():
                    _, last_values_tensor = self.network(
                        T.tensor(observations, dtype=T.float32, device=self.device))
                    last_values_ar: np.ndarray = last_values_tensor.cpu().numpy()

                self.train_network(last_values_ar)
                self.memory_buffer.reset()
                self.network.save_model(os.path.join("tmp", "network.pt"))
                T.save(self.network_optim, os.path.join(
                    "tmp", "network_optim.pt"))

            if (iteration * self.n_workers) % 1_000 < self.n_workers:
                total_duration = time.perf_counter()-t_start
                steps_done = self.n_workers*iteration

                score_mean = np.mean(s) if len(s) else 0
                score_std = np.std(s) if len(s) else 0
                score_err = 1.95 * score_std/(len(s)**0.5) if len(s) else 0
                fps = steps_done//total_duration
                print(f"**************************************************")
                print(
                    f"Iteration:      {iteration} of {int(total_iterations)}")
                print(f"learning rate:  {network_scheduler.get_last_lr()}")
                print(
                    f"Total Steps:    {steps_done} of {int(self.total_steps)}")
                print(f"Total duration: {total_duration:0.2f} seconds")
                print(f"Average Score:  {score_mean:0.2f} ± {score_err:0.2f}")
                print(f"Average FPS:    {fps}")

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
            probs, values = self.network(observation_tensor)
        action_sample = T.distributions.Categorical(probs)
        actions = action_sample.sample()
        log_probs: T.Tensor = action_sample.log_prob(actions)
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.squeeze(-1).cpu().numpy()

    def train_network(self, last_values: np.ndarray):
        # self.network.train

        observation_batches, action_batches, log_prob_batches, value_batches, reward_batches, terminal_batches = self.memory_buffer.sample()
        advantages_tensor: T.Tensor = self._calculate_advantages(
            reward_batches, value_batches, terminal_batches, last_values)

        batch_size = (self.step_size * self.n_workers) // self.n_batches
        for epoch in range(self.n_epochs):
            batch_starts = np.arange(
                0, self.n_workers*self.step_size, batch_size)
            indices = np.arange(self.n_workers*self.step_size)
            np.random.shuffle(indices)

            batches = [indices[i:i+batch_size] for i in batch_starts]

            observation_arr = observation_batches.reshape(
                (self.n_workers*self.step_size, *self.observation_shape))
            actions_arr = action_batches.reshape(
                (self.n_workers*self.step_size,))
            log_probs_arr = log_prob_batches.reshape(
                (self.n_workers*self.step_size,))
            values_arr = value_batches.reshape(
                (self.n_workers*self.step_size,))

            values_tensor = T.tensor(values_arr, device=self.device)

            for batch in batches:
                observation_tensor = T.tensor(
                    observation_arr[batch], dtype=T.float32, device=self.device)
                old_log_prob_tensor: T.Tensor = T.tensor(
                    log_probs_arr[batch], dtype=T.float32, device=self.device)
                actions_tensor = T.tensor(
                    actions_arr[batch], device=self.device)
                
                # normalized adv used in actor loss not value loss 
                norm_or_not_adv = advantages_tensor[batch].clone()
                if self.normalize_adv: 
                    norm_or_not_adv -= norm_or_not_adv.mean()
                    norm_or_not_adv /= (norm_or_not_adv.std()+1e-8)
                predicted_values: T.Tensor
                probs: T.Tensor
                probs, predicted_values = self.network(observation_tensor)
                predicted_values = predicted_values.squeeze()
                probs_dist = T.distributions.Categorical(probs)
                predicted_log_probs: T.Tensor = probs_dist.log_prob(
                    actions_tensor)

                predicted_probs = predicted_log_probs.exp()
                old_probs = old_log_prob_tensor.exp()

                dpp = ((old_probs-predicted_probs)**2).mean()

                prob_ratio = predicted_probs/old_probs

                weighted_prob_ratio = (
                    prob_ratio * norm_or_not_adv).mean()

                actor_loss = self.beta * dpp - weighted_prob_ratio

                returns = advantages_tensor[batch] + values_tensor[batch]
                critic_loss = 0.5*((returns - predicted_values)**2).mean()

                total_loss = actor_loss + 0.5 * critic_loss

                self.network.zero_grad()
                total_loss.backward()

                if self.max_grad_norm:
                    clip_grad_norm_(self.network.parameters(),
                                    self.max_grad_norm)
                self.network_optim.step()

    def _calculate_advantages(self, rewards: np.ndarray, values: np.ndarray, terminals: np.ndarray, last_values: np.ndarray):
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
        fig.savefig(os.path.join("tmp", "pop3d-"+file_name))
