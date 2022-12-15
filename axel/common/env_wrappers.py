import gym
from gym import Env
from gym.wrappers.frame_stack import FrameStack
from gym import spaces
import numpy as np
import torch as T
from torchvision import transforms as TR


class SkipFrame(gym.Wrapper):
    def __init__(self, env: Env, skip: int):
        '''
        Return only every 'skip'-th frame
        '''
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = T.tensor(observation.copy(), dtype=T.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = TR.Grayscale()
        observation: T.Tensor = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env: Env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.float32)

    def observation(self, observation):
        transform = TR.Compose([TR.Resize(self.shape), TR.Normalize(0, 255)])
        observation:T.Tensor = transform(observation).squeeze(0)
        arr:np.ndarray =observation.numpy()
        return arr


def apply_wrappers(env):
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    if gym.__version__ < "0.26":
        env = FrameStack(env, num_stack=4, new_step_api=True)
    else:
        env = FrameStack(env, num_stack=4)
    return env
