import gym
import time
import os
from collections import deque
from gym.vector import AsyncVectorEnv
import torch as T

import numpy as np
from torch.nn.utils.clip_grad import clip_grad_norm_
from typing import Callable, Sequence
from axel.common.networks import ImpalaEncoder, ImpalaRnnActorCriticNetwork, SmallRnnActorCriticNetwork
from axel.common.running_normalizer import RunningNormalizer
from axel.common.utils import calculate_explained_variance, seconds_to_HMS
from axel.ppo.memory_buffer import MemoryBuffer

class RecurrentPPO():
    def __init__(self,
                 game_fns: Sequence[Callable[[],gym.Env]],
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
                 ) -> None:
        self.vec_env = AsyncVectorEnv(game_fns)
        # self.network = ImpalaRnnActorCriticNetwork(self.vec_env.single_observation_space.shape,self.vec_env.single_action_space.n)
        self.network = SmallRnnActorCriticNetwork(self.vec_env.single_observation_space.shape,self.vec_env.single_action_space.n)
        self.encoder = self.network.encoder

        self.n_workers = self.vec_env.num_envs
        self.n_game_actions = self.vec_env.single_action_space.n
        self.observation_shape = self.vec_env.single_observation_space.shape


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
        self.reward_normalizer = RunningNormalizer(maxlen=100,gamma=gamma)

        self.optim = T.optim.Adam(self.network.parameters(),lr=self.lr)

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.network.to(self.device)

    def run(self):
        horizon = self.step_size
        t_start = time.perf_counter()
        device = self.device
        states,terminals = self.vec_env.reset()
        scaler = 1
        log_steps = 10000
        next_log = log_steps

        with T.no_grad():
            hts,cts = self.network.get_initials(self.n_workers)

        last_scores = deque(maxlen=50)
        workers_score = np.zeros((self.n_workers,))
        current_steps = 0
        min_steps = self.total_steps

        
        while current_steps< min_steps:
            all_states = []
            all_probs = []
            all_actions = []
            all_values = []
            all_terminals = []
            all_rewards = []
            all_hts = []
            all_cts = []
            for i in range(horizon):
                obs_t = T.tensor(states,dtype=T.float32,device=device)

                with T.no_grad():
                    probs_t , v_t , new_hts,new_cts = self.network.predict(obs_t,hts,cts)
                
                dist = T.distributions.Categorical(probs_t)
                actions_t = dist.sample()
                actions = actions_t.cpu().numpy()
                probs_ar = T.exp(dist.log_prob(actions_t)).cpu().numpy()
                values_ar = v_t.squeeze().cpu().numpy()

                z = self.vec_env.step(actions)
                current_steps += self.n_workers
                new_states , rewards,terminals,trucns,_ = z
                terminals:np.ndarray = np.logical_or(terminals,trucns)

                all_states.append(states)
                all_probs.append(probs_ar)
                all_actions.append(actions)
                all_values.append(values_ar)
                all_terminals.append(terminals)
                all_rewards.append(rewards)
                all_hts.append(hts)
                all_cts.append(cts)

                workers_score += rewards
                ar = np.argwhere(terminals)
                for idx in ar.squeeze(-1):
                    last_scores.append(workers_score[idx])
                workers_score[terminals] = 0

                states = new_states
                hts = new_hts
                cts = new_cts

                n_terminals = terminals.sum()
                with T.no_grad():
                    hts[terminals],cts[terminals] = self.network.get_initials(n_terminals)
            
            last_states_t = T.tensor(states,dtype=T.float32,device=device)
            with T.no_grad():
                _,last_v_t,_,_ = self.network.predict(last_states_t,hts,cts)
            last_values_ar = last_v_t.squeeze().cpu().numpy()

            all_states_ar = np.array(all_states,dtype=np.float32)
            all_probs_ar = np.array(all_probs,dtype=np.float32)
            all_actions_ar = np.array(all_actions,dtype=np.int32)
            all_values_ar = np.array(all_values,dtype=np.float32)
            all_terminals_ar = np.array(all_terminals,dtype=np.bool8)
            all_rewards_ar = np.array(all_rewards,dtype=np.float32)
            all_hts_t = T.stack(all_hts,dim=0)
            all_cts_t = T.stack(all_cts,dim=0)

            if self.normalize_rewards:
                rewards , scaler = self.reward_normalizer.normalize_rewards(all_rewards_ar,all_terminals_ar,self.max_reward_norm)
            
            all_adv_ar , all_returns_ar = self.calculate_adv(all_rewards_ar,all_values_ar,all_terminals_ar,last_values_ar)

            explained_variance = calculate_explained_variance(all_values_ar,all_returns_ar)
            if self.decay_lr:
                decay_rate = 0.1
                decay =  decay_rate ** (current_steps / min_steps)
                lr = self.lr * decay
                for g in self.optim.param_groups:
                    g['lr'] = lr
                self.optim.param_groups
            total_loss , actor_loss , critic_loss , entropy ,approx_kl= self.train_network(
                all_states_ar,all_actions_ar,all_probs_ar,all_adv_ar,all_returns_ar,all_terminals_ar,all_hts_t,all_cts_t)
            
            if current_steps >= next_log:
                next_log += log_steps
                duration = time.perf_counter() - t_start
                fps = current_steps // duration
                hours , minutes , seconds = seconds_to_HMS(duration)
                score_mean = np.mean(last_scores) if len(last_scores) > 0 else 0
                score_std = np.std(last_scores) if len(last_scores) > 0 else 0
                score_err = 1.95 * score_std / \
                        (len(last_scores)**0.5) if len(last_scores) > 0 else 0

                print(f"*" * 20)
                print(f"[INFO] Step: {current_steps+1} of {min_steps}")
                print(f"[INFO] Duration: {hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}")
                print(f"[INFO] FPS: {fps}")
                print(f"[INFO] Learning Rate: {self.optim.param_groups[0]['lr']:0.3e}")
                print(f"[INFO] Average Score: {score_mean:0.3f} ± {score_err:0.3f}")
                print(f"[INFO] Scaler: {scaler:0.3f}")
                print(f"[INFO] Total Loss: {total_loss :0.3f}")
                print(f"[INFO] Actor Loss: {actor_loss :0.3f}")
                print(f"[INFO] Critic Loss: {critic_loss :0.3f}")
                print(f"[INFO] Entropy: {entropy :0.3f}")
                print(f"[INFO] Explained Variance: {explained_variance :0.3f}")
                print(f"[INFO] Approx KL: {approx_kl :0.3f}")

            T.save(
                {"network" : self.network.state_dict(),
                "reward_normalizer" : self.reward_normalizer,
                "optimizer" : self.optim
                },os.path.join("tmp","ppo_recurrent.pt"))
            self.network.save_model(os.path.join("tmp","ppo_recurrent_network.pt"))
        
    def calculate_adv(
            self,
            rewards:np.ndarray,
            values:np.ndarray,
            terminals:np.ndarray,
            last_values:np.ndarray):
        horizon = len(values)
        advs = np.zeros_like(values)
        returns = np.zeros_like(values)
        next_val = last_values.copy()
        next_adv = np.zeros_like(last_values)
        gamma = self.gamma
        gae_lam = self.gae_lam

        for t in reversed(range(horizon)):
            current_terminals :np.ndarray= terminals[t]
            current_rewards :np.ndarray= rewards[t]
            current_values :np.ndarray= values[t]
            next_val[current_terminals] = 0
            next_adv[current_terminals] = 0

            delta = current_rewards + gamma * next_val - current_values
            current_adv = delta + gamma * gae_lam * next_adv
            next_val = current_values.copy()
            next_adv = current_adv.copy()
            
            advs[t] = current_adv
            returns[t] = current_adv + current_values
        
        return advs,returns

    def train_network(self,
                  observations:np.ndarray,
                  actions:np.ndarray,
                  action_probs:np.ndarray,
                  advantages:np.ndarray,
                  returns:np.ndarray,
                  terminals:np.ndarray,
                  hts:T.Tensor,
                  cts:T.Tensor):
        n_epochs = self.n_epochs
        n_batches = self.n_batches
        sample_size = len(observations)
        assert sample_size % n_batches == 0
        batch_size = sample_size // n_batches
        device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        observations_t = T.tensor(observations,device=device,dtype=T.float32)
        actions_t = T.tensor(actions,device=device,dtype=T.int32)
        actions_probs_t = T.tensor(action_probs,device=device,dtype=T.float32)
        adv_t = T.tensor(advantages,device=device,dtype=T.float32)
        if self.normalize_adv:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        returns_t = T.tensor(returns,device=device,dtype=T.float32)
        total_losses = []
        actor_losses = []
        critic_losses = []
        entropies = []
        # approx_kl_t = T.zeros((1,),device=self.device)
        approx_kls = []
        
        for epoch in range(n_epochs):
            batch_starts = np.arange(0,sample_size,batch_size)
            indices = np.arange(0,sample_size)
            batches = [indices[i:i+batch_size] for i in batch_starts]

            for batch in batches:
                observations_batch = observations_t[batch]
                old_probs_batch = actions_probs_t[batch]
                actions_batch = actions_t[batch]
                adv_batch = adv_t[batch].clone()
                returns_batch = returns_t[batch]
                hts_batch = hts[batch]
                cts_batch = cts[batch]
                terminals_batch = terminals[batch]
                all_ht = []
                all_ct = []
                new_probs_batch = []
                new_vs_batch = []
                ht,ct = hts_batch[0],cts_batch[0]
                xs = self.network.encode(observations_batch)
                for t,x in enumerate(xs):
                    all_ht.append(ht)
                    all_ct.append(ct)
                    new_probs,v,ht,ct = self.network.single(x,ht,ct)
                    new_probs_batch.append(new_probs)
                    new_vs_batch.append(v)
                    current_terminals:np.ndarray = terminals_batch[t]
                    ht_masks = T.ones_like(ht)
                    ht_masks[current_terminals] = 0
                    ct_masks = T.ones_like(ct)
                    ct_masks[current_terminals] = 0
                    ht0_masks = T.ones_like(ht)
                    ct0_masks = T.ones_like(ct)
                    ht0_masks[current_terminals == False] = 0
                    ct0_masks[current_terminals == False] = 0
                    ht0,ct0 = self.network.get_initials(len(current_terminals))
                    ht = ht * ht_masks + ht0 * ht0_masks
                    ct = ct * ct_masks + ct0 * ct0_masks
                new_probs = T.stack(new_probs_batch,dim=0)
                v = T.stack(new_vs_batch,dim=0)
                v = v.squeeze()
                dist = T.distributions.Categorical(new_probs)
                entropy:T.Tensor = dist.entropy().mean()

                # actor loss
                new_log_probs = dist.log_prob(actions_batch)
                old_log_probs = old_probs_batch.log()
                prob_ratio :T.Tensor = (new_log_probs-old_log_probs).exp()
                weighted_prob = adv_batch * prob_ratio
                clipped_ratio = prob_ratio.clamp(1-self.policy_clip,1+self.policy_clip)
                weighted_clipped_prob = clipped_ratio * adv_batch
                actor_loss = -T.min(weighted_prob,weighted_clipped_prob).mean()
                approx_kl_t = ((prob_ratio - 1) - T.log(prob_ratio)).mean()
                approx_kls.append(approx_kl_t)

                # critic loss
                critic_loss = (0.5*(returns_batch - v)**2).mean()

                total_loss:T.Tensor = actor_loss + self.critic_coef*critic_loss * 0.5 - self.entropy_coef * entropy


                self.optim.zero_grad()

                total_loss.backward()

                if self.max_grad_norm:
                    clip_grad_norm_(self.network.parameters(),self.max_grad_norm)
                self.optim.step()

                entropies.append(entropy)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                total_losses.append(total_loss)


        with T.no_grad():
            mean_total_loss = T.stack(total_losses).mean().cpu().item()
            mean_critic_loss = T.stack(critic_losses).mean().cpu().item()
            mean_actor_loss = T.stack(actor_losses).mean().cpu().item()
            mean_entropy = T.stack(entropies).mean().cpu().item()
            approx_kl = T.stack(approx_kls).mean().cpu().item()
        return mean_total_loss,mean_actor_loss,mean_critic_loss,mean_entropy,approx_kl


    def __del__(self):
        self.vec_env.close()

            
