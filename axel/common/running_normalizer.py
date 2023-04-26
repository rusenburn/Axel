from collections import deque
import numpy as np

class RunningNormalizer():
    def __init__(self,maxlen:int=100,gamma:float=0.99) -> None:
        self.sum_returns = deque(maxlen=maxlen)
        self.sum_squared_returns = deque(maxlen=maxlen)
        self.counts = deque(maxlen=maxlen)
        self.last_values:np.ndarray|None = None
        self.gamma = gamma
    
    def normalize_rewards(self,rewards:np.ndarray,terminals:np.ndarray,clip:float):
        if self.last_values is None:
            self.last_values = np.zeros_like(rewards[0])
        
        steps = len(rewards)
        reveresed_returns = np.zeros_like(rewards)

        for t in range(steps):
            current_reward:np.ndarray = rewards[t]
            next_value:np.ndarray = self.last_values
            current_value:np.ndarray = current_reward + self.gamma * next_value
            reveresed_returns[t] = current_value.copy()
            self.last_values = current_value.copy()
            self.last_values[terminals[t]] = 0
        
        sum_ = reveresed_returns.sum()
        sum_sqaured = np.sum(reveresed_returns**2)
        count = reveresed_returns.size

        self.sum_returns.append(sum_)
        self.sum_squared_returns.append(sum_sqaured)
        self.counts.append(count)

        total_sum = np.sum(self.sum_returns)
        total_squared_sum = np.sum(self.sum_squared_returns)
        total_count = np.sum(self.counts)

        variace = (total_squared_sum / total_count) - (total_sum/total_count)**2
        std:float = variace**0.5

        rewards = rewards/(std+1e-8)
        rewards = rewards.clip(-clip,clip)
        return rewards , std
