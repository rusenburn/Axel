from abc import ABC, abstractmethod
import torch as T
import torch.nn as nn
import numpy as np


class NetworkBase(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def save_model(self, path: str | None) -> None:
        '''
        Saves the model into path
        '''
        raise NotImplementedError("Calling abstract save_model")

    @abstractmethod
    def load_model(self, path: str | None) -> None:
        '''
        Loads model from path
        '''
        raise NotImplementedError("Calling abstract load_model")


class PytorchNetwork(nn.Module, NetworkBase):
    def __init__(self) -> None:
        super().__init__()
        self.last_path = None

    def save_model(self, path: str | None) -> None:
        if path:
            self.last_path = path
        try:
            T.save(self.state_dict(), path)
        except:
            print(f'could not save nn to {self.last_path}')

    def load_model(self, path: str | None) -> None:
        if path:
            self.last_path = path
        try:
            self.load_state_dict(T.load(self.last_path))
            print(f'The nn was loaded from {self.last_path}')
        except:
            print(f'could not load nn from {self.last_path}')


class ActorNetwork(PytorchNetwork):
    def __init__(self, observation_space: np.ndarray, n_actions: int) -> None:
        super().__init__()

        self._probs = nn.Sequential(
            # (?,4,84,84)
            nn.Conv2d(observation_space[0], 32, 8, 4),
            # (?,32,20,20)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            # (?,64,9,9)
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            # (?,64,7,7)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, observation: T.Tensor) -> T.Tensor:
        probs: T.Tensor = self._probs(observation)
        return probs


class CriticNetwork(PytorchNetwork):
    def __init__(self, observation_space: np.ndarray) -> None:
        super().__init__()
        self._v = nn.Sequential(
            nn.Conv2d(observation_space[0], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, 1))

    def forward(self, observation: T.Tensor) -> T.Tensor:
        v: T.Tensor = self._v(observation)
        return v


class ActorCriticNetwork(PytorchNetwork):
    def __init__(self, observation_space: np.ndarray, n_actions: int) -> None:
        super().__init__()
        self._shared = nn.Sequential(
            nn.Conv2d(observation_space[0], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 512),
            nn.ReLU()
        )

        self._probs = nn.Sequential(
            nn.Linear(512, n_actions),
            nn.Softmax(dim=-1)
        )
        self._v = nn.Linear(512, 1)

    def forward(self, observation: T.Tensor) -> tuple[T.Tensor, T.Tensor]:
        shared: T.Tensor = self._shared(observation)
        probs: T.Tensor = self._probs(shared)
        v: T.Tensor = self._v(shared)
        return probs, v
