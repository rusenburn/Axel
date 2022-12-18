import time
import numpy as np
import torch as T
import gym.vector
import os
from collections import deque
from torch.optim import Adam
from torch.distributions import Categorical
from torch.nn.utils.clip_grad import clip_grad_norm_
from .memory_buffer import MemoryBuffer
from ..common.networks import NetworkBase, ActorNetwork, CriticNetwork


class Ppo:
    def __init__(self, vec_env: gym.vector.VectorEnv,
                 total_steps=1e5,
                 step_size=5, n_batches=4, n_epochs=4,
                 gamma=0.99, gae_lam=0.95, policy_clip=0.2,
                 lr=2.5e-4) -> None:
        self.vec_env = vec_env
        self.n_workers = vec_env.num_envs
        assert (self.n_workers*step_size) % n_batches == 0
        self.total_steps = total_steps
        self.step_size = step_size
        self.n_batches = n_batches
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lam = gae_lam
        self.policy_clip = policy_clip
        self.lr = lr
        self.memory_buffer = MemoryBuffer(
            self.vec_env.single_observation_space.shape, self.vec_env.num_envs, self.step_size)

        self.max_grad_norm = 0.5
        self.n_game_actions = self.vec_env.single_action_space.n
        self.actor = ActorNetwork(
            self.vec_env.single_observation_space.shape, self.n_game_actions)

        self.critic = CriticNetwork(
            self.vec_env.single_observation_space.shape)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr, eps=1e-5)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr, eps=1e-5)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.actor.to(self.device)
        self.critic.to(self.device)

    def run(self) -> None:
        t_start = time.perf_counter()

        observations: np.ndarray
        infos: dict
        observations, infos = self.vec_env.reset()
        total_scores = np.zeros((self.n_workers,), dtype=np.float32)
        s = deque([], 100)

        total_iterations = int(self.total_steps//self.n_workers)
        def lr_fn(x): return (total_iterations-x) / total_iterations

        actor_scheduler = T.optim.lr_scheduler.LambdaLR(
            self.actor_optim, lr_fn)
        critic_shceduler = T.optim.lr_scheduler.LambdaLR(
            self.critic_optim, lr_fn)
        for iteration in range(1, int(self.total_steps//self.n_workers)+1):
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
                with T.no_grad():
                    last_values_t: T.Tensor = self.critic(
                        T.tensor(observations.copy(), dtype=T.float32, device=self.device))
                    last_values_ar: np.ndarray = last_values_t.cpu().numpy()
                self.train_network(last_values_ar)
                self.memory_buffer.reset()
                self.actor.save_model(os.path.join("tmp", "actor.pt"))
                self.critic.save_model(os.path.join("tmp", "critic.pt"))
                T.save(self.actor_optim, os.path.join("tmp", "actor_optim.pt"))
                T.save(self.critic_optim, os.path.join(
                    "tmp", "critic_optim.pt"))

            if (iteration * self.n_workers) % 500 == 0:
                # TODO LOG
                iteration_duration = time.perf_counter()-iteration_start
                total_duration = time.perf_counter()-t_start
                iteration_fps = self.n_workers // iteration_duration

                print(f"**************************************************")
                print(
                    f"Iteration:              {iteration} of {int(self.total_steps//self.n_workers)}")
                print(
                    f"learning rate:          {actor_scheduler.get_last_lr()}")
                print(
                    f"Total Steps:            {self.n_workers*iteration} of {int(self.total_steps)}")
                print(
                    f"Average Score:          {np.mean(s) if len(s)>0 else 0:0.2f}")
                print(
                    f"Iteration duration:     {iteration_duration:0.2f} seconds")
                print(f"Iteration FPS:          {iteration_fps}")
                print(f"Total duration:         {total_duration:0.2f} seconds")
                print(
                    f"Average FPS:            {self.n_workers * iteration//total_duration }")
            actor_scheduler.step()
            critic_shceduler.step()
            # self.actor.save_model(os.path.join("tmp", "actor.pt"))
            # self.critic.save_model(os.path.join("tmp", "critic.pt"))
            # T.save(self.actor_optim, os.path.join("tmp", "actor_optim.pt"))
            # T.save(self.critic_optim, os.path.join("tmp", "critic_optim.pt"))

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
        for epoch in range(self.n_epochs):
            state_batches, action_batches, log_prob_batches, \
                value_batches, reward_batches, terminal_batches = self.memory_buffer.sample()

            # advantages_arr = np.zeros(
            #     (self.n_workers, self.step_size), dtype=np.float32)
            # for worker in range(self.n_workers):
            #     for step in range(self.step_size-1):
            #         discount = 1
            #         a_t = 0
            #         for k in range(step, self.step_size-1):
            #             current_reward = reward_batches[worker][k]
            #             current_val = value_batches[worker][k]
            #             next_val = value_batches[worker][k+1]

            #             a_t += discount*(current_reward+self.gamma*next_val *
            #                              (1-int(terminal_batches[worker][k]))-current_val)
            #             discount *= self.gamma*self.gae_lam
            #         advantages_arr[worker][step] = a_t
            # advantages_tensor = T.tensor(
            #     advantages_arr.flatten(), dtype=T.float32, device=self.device)

            advantages_tensor = self._calculate_advatages(
                reward_batches, value_batches, terminal_batches, last_values)
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
            for batch in batches:
                states_tensor: T.Tensor = T.tensor(
                    states_arr[batch], dtype=T.float32, device=self.device)
                old_log_probs_tensor: T.Tensor = T.tensor(
                    log_probs_arr[batch], dtype=T.float32, device=self.device)
                actions_tensor: T.Tensor = T.tensor(
                    actions_arr[batch], device=self.device)
                # Normalize advantages
                normalized_advatages = advantages_tensor[batch]
                normalized_advatages = (
                    normalized_advatages - normalized_advatages.mean())/normalized_advatages.std()

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
                # weighted_clipped_probs *= normalized_advatages
                actor_loss = -T.min(weighted_probs,
                                    weighted_clipped_probs).mean()

                # find critic loss
                returns = advantages_tensor[batch] + values_tensor[batch]
                old_values_predictions = values_tensor[batch]
                clipped_values_predictions = old_values_predictions + \
                    T.clamp(critic_values-old_values_predictions, -
                            self.policy_clip, self.policy_clip)
                critic_loss = (returns - critic_values)**2
                # critic_loss = 0.5*critic_loss.mean()
                critic_loss_2 = (returns - clipped_values_predictions)**2
                critic_loss = 0.5*T.max(critic_loss, critic_loss_2).mean()
                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

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

    def _calculate_advatages(self, rewards: np.ndarray, values: np.ndarray, terminals: np.ndarray, last_values: np.ndarray):
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

        # try normalize/standrize advantages
        # advantages = (advantages-advantages.mean()) / (advantages.std()+1e-8)
        return advantages
