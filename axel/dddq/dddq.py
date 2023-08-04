import os
import torch
import numpy as np
import gym
import time
from typing import Callable , Sequence
from collections import deque
from  gym.vector.async_vector_env import AsyncVectorEnv
from  gym.vector.sync_vector_env import SyncVectorEnv


from axel.common.networks import DuelingQNetwork
from axel.common.per import PER
from axel.common.utils import seconds_to_HMS

class DDDQ():
    def __init__(self,
                 env_fns: Sequence[Callable[[],gym.Env]],
                 learning_rate=2.5e-4,
                 total_episodes=1_000_000,
                 batch_size=64,
                 max_tau= 10000,
                 exploration_start = 1,
                 exploration_end = 0.01,
                 decay_rate = 5e-5,
                 gamma = 0.95,
                 pretrain_length=1_000,
                 memory_size=100_000,
                 name="DDDQ",
                 load_name="",
                 ) -> None:
        
        self.vec_env = SyncVectorEnv(env_fns)
        self.n_envs = self.vec_env.num_envs
        self.learning_rate = learning_rate
        self.total_steps = total_episodes
        self.batch_size= batch_size
        self.max_tau = max_tau
        self.gamma_avg = 0.001
        self.exploration_start = exploration_start
        self.exploration_end = exploration_end
        self.decay_rate = decay_rate
        self.gamma = gamma
        self.pretrain_length = pretrain_length
        self.memory_size = memory_size
        self.name = name
        self.load_name = load_name

        self.n_actions = self.vec_env.single_action_space.n
        self.observation_shape = self.vec_env.single_observation_space.shape
        self.observation_dtype = self.vec_env.single_observation_space.dtype
        self.device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu")
        self.network = DuelingQNetwork(self.observation_shape,self.n_actions)
        self.target = DuelingQNetwork(self.observation_shape,self.n_actions)
        self.network.to(self.device)
        self.target.to(self.device)
        self.network_optimizer = torch.optim.Adam(self.network.parameters(),self.learning_rate)
        self.memory = PER(memory_size)


    def run(self)->None:
        states ,infos = self.vec_env.reset()
        t_start = time.perf_counter()

        tau = 0
        decay_step = 0
        target_version = 0
        self.update_target_network()

        if self.load_name:
            load_data : dict = torch.load(os.path.join("tmp",f"{self.load_name}.dddq.data.pt"))
            self.network.load_state_dict(load_data["network"])
            self.target.load_state_dict(load_data["target"])
            self.network_optimizer.load_state_dict(load_data["optimizer"])
            decay_step = load_data["decay_step"]
            tau = load_data["tau"]
            del load_data

        print(f"[info] Filling replay with {self.pretrain_length} training examples")

        while self.memory.current_capacity < self.pretrain_length:
            actions , _ = self.choose_actions(states,decay_step)
            new_states , rewards , dones , truncs , infos = self.vec_env.step(actions)
            dones = np.logical_or(dones,truncs)
            experiences = [(state,action , reward,new_state,done) for state,action,reward,new_state,done in zip(states,actions,rewards,new_states,dones)]
            for experience in experiences:
                self.memory.store(experience)
            states = new_states
        steps = 0
        next_log = 0
        log_intervals = 500
        next_log +=log_intervals
        episode_rewards = np.zeros((self.n_envs,),dtype=np.float32)
        last_rewards = deque(maxlen=50)

        states , infos = self.vec_env.reset()
        t_start = time.perf_counter()
        print(f"[info] Training started")
        while steps < self.total_steps:
            actions , explore_probability = self.choose_actions(states,decay_step)
            new_states, rewards,dones,truncs,infos = self.vec_env.step(actions)
            dones = np.logical_or(dones,truncs)
            episode_rewards += rewards
            for i,d in enumerate(dones):
                if d:
                    last_rewards.append(episode_rewards[i])
                    episode_rewards[i] = 0
            experiences = [(state,action , reward,new_state,done) for state,action,reward,new_state,done in zip(states,actions,rewards,new_states,dones)]
            for experience in experiences:
                self.memory.store(experience)
            states = new_states
            steps+=self.n_envs
            decay_step+=self.n_envs
            tree_idx , batch,is_weights_mb = self.memory.sample(self.batch_size)

            states_mb = np.array([ex[0] for ex in batch])
            actions_mb = np.array([ex[1] for ex in batch])
            rewards_mb = np.array([ex[2] for ex in batch])
            new_states_mb = np.array([ex[3] for ex in batch])
            dones_mb = np.array([ex[4] for ex in batch])

            target_qs_batch = []

            with torch.no_grad():
                new_states_t = torch.tensor(new_states_mb,dtype=torch.float32,device=self.device)
                q_target_next = self.target.evaluate(new_states_t/255.0).cpu().numpy()
                q_next_state = self.network.evaluate(new_states_t/255.0).cpu().numpy()

            for i in range(len(batch)):
                terminal = dones_mb[i]

                new_action = np.argmax(q_next_state[i],axis=-1)

                if terminal:
                    target_qs_batch.append(rewards_mb[i])
                else:
                    target = rewards_mb[i] + self.gamma* q_target_next[i][new_action]
                    target_qs_batch.append(target)
            
            target_mb = np.array([ex for ex in target_qs_batch])

            target_q_t = torch.tensor(target_mb,dtype=torch.float32,device=self.device)
            states_mb_t = torch.tensor(states_mb,dtype=torch.float32,device=self.device)
            actions_mb_t = torch.tensor(actions_mb,dtype=torch.int64,device=self.device)
            is_weights_mb_t = torch.tensor(is_weights_mb,dtype=torch.float32,device=self.device)
            qs = self.network.evaluate(states_mb_t/255.0)
            qsa = qs[torch.arange(0,len(batch)),actions_mb_t]
            with torch.no_grad():
                absolute_errors_t = torch.abs(target_q_t-qsa).squeeze()
            loss = (is_weights_mb_t *(target_q_t - qsa)**2).mean()

            self.network_optimizer.zero_grad()
            loss.backward()
            self.network_optimizer.step()

            training_loss = loss.detach().cpu().item() 

            absolute_errors = absolute_errors_t.cpu().numpy()

            self.memory.batch_update(tree_idx,absolute_errors)
            tau += 1

            if tau >= self.max_tau:
                tau = 0
                self.update_target_network()
                target_version+=1
                print(f"[info] updating target network")

            if steps > next_log:
                next_log += log_intervals
                reward_mean = np.mean(last_rewards)
                t_now = time.perf_counter()
                duration = t_now-t_start
                fps = duration / steps
                hours , minutes , seconds = seconds_to_HMS(duration)
                print(f"*" * 10)
                print(f"[info] Steps {steps} ,FPS {fps:0.1f} , Reward mean {reward_mean:0.3f}")
                print(f"[info] Loss {training_loss:0.3f} , Explore Probability {explore_probability:0.3f}")
                print(f"[info] Replay Capacity {self.memory.current_capacity}")
                print(f"[info] Duration: {hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}")
                print(f"[info] Target: {target_version} , Tau {tau} {tau*100/self.max_tau:0.1f}%")
                print(f"[info] Saving data...")
                save_data = {
                    "network" : self.network.state_dict(),
                    "target" : self.target.state_dict(),
                    "optimizer" : self.network_optimizer.state_dict(),
                    "decay_step" : decay_step,
                    "tau" : tau,
                }
                torch.save(save_data,os.path.join("tmp",f"{self.name}.dddq.data.pt"))

    def choose_actions(self,observations:np.ndarray,decay_step:int):
        explore_probability = self.exploration_end +  (self.exploration_start - self.exploration_end)  * np.exp(-self.decay_rate * decay_step)
        exp_exp_tradeoff = np.random.rand(self.n_envs)
        actions = np.random.choice(self.n_actions,size=self.n_envs,replace=True)
        explore_predicate = explore_probability > exp_exp_tradeoff
        exploit_predicate = explore_predicate == False

        if np.any(exploit_predicate):
            observations_t = torch.tensor(observations[exploit_predicate],dtype=torch.float32,device=self.device)
            with torch.no_grad():
                qs = self.network.evaluate(observations_t/255.0)
            choice:np.ndarray = qs.argmax(dim=-1).cpu().numpy()
            choice = choice.squeeze()
            actions[exploit_predicate] = choice
        return actions , explore_probability

    def update_target_network(self):
        self.target.load_state_dict(self.network.state_dict())
        # with torch.no_grad():
        #     for target_p, network_p in zip(self.target.parameters(), self.network.parameters()):
        #         target_p.data.copy_(self.gamma_avg * network_p.data +
        #                        (1-self.gamma_avg)*target_p.data)

