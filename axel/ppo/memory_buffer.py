import numpy as np

class MemoryBuffer:
    def __init__(self,observation_shape:tuple,n_workers:int,worker_steps:int) -> None:
        self.observation_shape = observation_shape
        self.n_workers = n_workers
        self.worker_steps = worker_steps
        self.reset()
    
    def reset(self):
        self.current_worker = 0
        self.current_step = 0
        self.observations = np.zeros((self.worker_steps,self.n_workers,*self.observation_shape),dtype=np.float32)
        self.actions = np.zeros((self.worker_steps,self.n_workers),dtype=np.float32)
        self.rewards = np.zeros((self.worker_steps,self.n_workers),dtype=np.float32)
        self.log_probs = np.zeros((self.worker_steps,self.n_workers),dtype=np.float32)
        self.values = np.zeros((self.worker_steps,self.n_workers),dtype=np.float32)
        self.terminals = np.zeros((self.worker_steps,self.n_workers),dtype=np.bool8)

    def save(self,step_observations:np.ndarray,step_actions:np.ndarray,step_log_probs:np.ndarray,step_values:np.ndarray,step_rewards:np.ndarray,step_terminals:np.ndarray):
        self.observations[self.current_step] = np.array(step_observations)
        self.actions[self.current_step] = step_actions
        self.log_probs[self.current_step] = step_log_probs
        self.rewards[self.current_step] = step_rewards
        self.values[self.current_step] = step_values
        self.terminals[self.current_step] = step_terminals
        self.current_step+=1


    def sample(self)->tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.observations.swapaxes(0,1),self.actions.swapaxes(0,1),self.log_probs.swapaxes(0,1),self.values.swapaxes(0,1),self.rewards.swapaxes(0,1),self.terminals.swapaxes(0,1)


    def _incerement_indices(self):
        if self.current_worker == self.n_workers-1:
            self.current_worker = 0
            if self.current_step == self.worker_steps-1:
                self.current_step = 0
            else:
                self.current_step += 1
        else:
            self.current_worker += 1
