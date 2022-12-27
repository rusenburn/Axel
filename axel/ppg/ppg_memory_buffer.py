import numpy as np
from ..ppo.memory_buffer import MemoryBuffer

class PpgMemoryBuffer():
    def __init__(self, observation_shape: tuple, n_workers: int, worker_steps: int,n_phase_iterations:int) -> None:
        self.ppo_memory_buffer = MemoryBuffer(observation_shape, n_workers, worker_steps)
        self.observation_shape = observation_shape
        self.n_workers = n_workers
        self.worker_steps = worker_steps
        self.n_phase_iterations = n_phase_iterations
        self.n_examples = self.n_workers * self.worker_steps * self.n_phase_iterations
        self.reset_aux_data()
    


    def reset_ppo_data(self):
        return self.ppo_memory_buffer.reset()
    
    def save_ppo_data(self, step_observations: np.ndarray, step_actions: np.ndarray, step_log_probs: np.ndarray, step_values: np.ndarray, step_rewards: np.ndarray, step_terminals: np.ndarray):
        return self.ppo_memory_buffer.save(step_observations, step_actions, step_log_probs, step_values, step_rewards, step_terminals)
    
    def sample_ppo_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.ppo_memory_buffer.sample()

    def reset_aux_data(self):
        self.aux_observations = np.zeros((self.n_examples,*self.observation_shape),dtype=np.float32)
        self.aux_returns = np.zeros((self.n_examples,),dtype=np.float32)
        self.aux_index = 0

    def append_aux_data(self,observations:np.ndarray,returns:np.ndarray):
        if self.aux_index >= self.n_examples:
            raise ValueError("Cannot append data to full sized buffer try using reset_aux_data() method")
        n_iteration_examples = self.n_workers*self.worker_steps
        begin = self.aux_index* n_iteration_examples
        end = begin + n_iteration_examples
        self.aux_observations[begin:end] = observations.copy()
        self.aux_returns[begin:end] = returns.copy()
        self.aux_index+=1
    
    def sample_aux_data(self)->tuple[np.ndarray,np.ndarray]:
        return self.aux_observations,self.aux_returns





        