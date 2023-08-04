from abc import ABC, abstractmethod
import axel.common.torch_utils as tu
from typing import Sequence
import torch as T
from torch.nn import functional as F
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
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def act(self,observation:T.Tensor)->T.Tensor:
        raise NotImplementedError()

class CriticNetwork(PytorchNetwork):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def evaluate(self,observation:T.Tensor)->T.Tensor:
        raise NotImplementedError()

class ActorCriticNetwork(ActorNetwork,CriticNetwork):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def act_and_eval(self,observation:T.Tensor)->tuple[T.Tensor,T.Tensor]:
        raise NotImplementedError()

class StateActionNetwork(PytorchNetwork):
    def __init__(self) -> None:
        super().__init__()
    
    def evaluate(self,observation:T.Tensor)->T.Tensor:
        raise NotImplementedError()

class CnnActorNetwork(ActorNetwork):
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

    def act(self, observation: T.Tensor) -> T.Tensor:
        return self(observation)


class CnnCriticNetwork(CriticNetwork):
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

    def evaluate(self, observation: T.Tensor) -> T.Tensor:
        return self(observation)


class CnnActorCriticNetwork(ActorCriticNetwork):
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

    def act(self, observation: T.Tensor) -> T.Tensor:
        prob,_ = self(observation)
        return prob
    
    def evaluate(self, observation: T.Tensor) -> T.Tensor:
        _,v = self(observation)
        return v

    def act_and_eval(self, observation: T.Tensor) -> tuple[T.Tensor, T.Tensor]:
        prob,v = self(observation)
        return prob,v


class SmallActorNetwork(ActorNetwork):
    def __init__(self, n_actions, input_dims,fc1_dims=256, fc2_dims=256):
        super(SmallActorNetwork, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        probs = self.actor(state)
        return probs

    def act(self, observation: T.Tensor) -> T.Tensor:
        return self(observation)

class SmallCriticNetwork(CriticNetwork):
    def __init__(self, input_dims, fc1_dims=256, fc2_dims=256):
        super(SmallCriticNetwork, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value
    
    def evaluate(self, observation: T.Tensor) -> T.Tensor:
        return self(observation)

class SmallActorCriticNetwork(ActorCriticNetwork):
    def __init__(self, n_actions, input_dims,fc1_dims=256, fc2_dims=256):
        super(SmallActorCriticNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state)->tuple[T.Tensor,T.Tensor]:
        dist = self.actor(state)
        val = self.critic(state)
        return dist,val

    def act(self, observation: T.Tensor) -> T.Tensor:
        prob,_ =  self(observation)
        return prob
    def evaluate(self, observation: T.Tensor) -> T.Tensor:
        _,v =  self(observation)
        return v
    def act_and_eval(self, observation: T.Tensor) -> tuple[T.Tensor, T.Tensor]:
        prob,v = self(observation)
        return prob,v

def sequential(layers, x, *args, diag_name=None):
    for (i, layer) in enumerate(layers):
        x = layer(x, *args)
    return x

class CnnBasicBlock(nn.Module):
    def __init__(self,inchan,scale=1,batchnorm=False) -> None:
        super().__init__()
        self.inchan = inchan
        self.batchnorm = batchnorm
        s = scale**0.5
        self.conv0 = tu.NormConv2d(T.nn.Conv2d(self.inchan,self.inchan,3,padding=1),scale=s)
        self.conv1 = tu.NormConv2d(T.nn.Conv2d(self.inchan,self.inchan,3,padding=1),scale=s)
        if self.batchnorm:
            self.bn0 = nn.BatchNorm2d(self.inchan)
            self.bn1 = nn.BatchNorm2d(self.inchan)
        self.res  = self._res(batchnorm)
        
    def _res(self,batchnorm:bool):
        seq = nn.Sequential()
        if batchnorm:
            seq.append(self.bn0)
        seq.append(nn.ReLU())
        seq.append(self.conv0)
        if batchnorm:
            seq.append(self.bn1)
        seq.append(nn.ReLU(inplace=True))
        seq.append(self.conv1)
        return seq
    def forward(self,x:T.Tensor):
        return x + self.res(x)

class CnnDownStack(nn.Module):
    def __init__(self,inchan,nblocks,outchan,scale=1,pool=True,**kwargs) -> None:
        super().__init__()

        self.inchan = inchan
        self.outchan=outchan
        self.pool = pool

        s = scale/(nblocks**0.5)
        self.first = nn.Sequential(
            tu.NormConv2d(nn.Conv2d(inchan,outchan,3,padding=1)))
        if self.pool:
            self.first.append(nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
        self.blocks = nn.Sequential(
            *[CnnBasicBlock(outchan,scale=s,**kwargs) for _ in range(nblocks)]
        )

    def forward(self,x):
        x = self.first(x)
        x = self.blocks(x)
        return x

    def output_shape(self,inshape):
        c,h,w = inshape

        assert c == self.inchan
        if getattr(self,"pool",True):
            return (self.outchan,(h+1)//2,(w+1)//2)
        else:
            return (self.outchan,h,w)

class ImpalaCNN(nn.Module):
    name = "ImpalaCNN"
    def __init__(self,inshape,chans,outsize,scale_ob,nblock,final_relu=True,**kwargs) -> None:
        super().__init__()
        self.scale_ob = scale_ob
        c,h,w = inshape
        curshape = (c,h,w)

        s = 1/((len(chans))**0.5)
        self.stacks = nn.ModuleList()
        for outchan in chans:
            stack = CnnDownStack(curshape[0],nblocks=nblock,outchan=outchan,scale=s,**kwargs)
            self.stacks.append(stack)
            curshape = stack.output_shape(curshape)
        self.seq_stacks = nn.Sequential(*self.stacks)
        self.dense = tu.NormLinear(nn.Linear(tu.intprod(curshape),outsize),scale=1.4)
        self.outsize = outsize
        self.final_relu = final_relu
    
    def forward(self,x:T.Tensor):
        x = x.to(T.float32) / self.scale_ob

        b = x.shape[:-3]
        o = x.shape[-3:]
        o = [*b,*o]
        x = x.reshape(*o)
        # x = sequential(self.stacks,x,diag_name=self.name)
        x = self.seq_stacks(x)
        x = x.reshape(*b,*x.shape[1:])
        *batch_shape , h,w,c = x.shape 
        x = x.reshape(*batch_shape,h*w*c)
        x = T.relu(x)
        x = self.dense(x)
        if self.final_relu:
            x = T.relu(x)
        return x

class ImpalaEncoder(nn.Module):
    def __init__(self,inshape,outsize=256,chans=(16,32,32),scale_ob=255.0,nblocks=2,**kwargs) -> None:
        super().__init__()
        self.cnn = ImpalaCNN(
            inshape=inshape,
            chans=chans,
            scale_ob=scale_ob,
            nblock=nblocks,
            outsize=outsize,
            **kwargs
        )
    
    def forward(self,x):
        x = self.cnn(x)
        return x

class ImpalaActorCritic(ActorCriticNetwork):
    def __init__(self,observations_shape,n_actions:int) -> None:
        super().__init__()
        outsize = 256
        self.encoder = ImpalaEncoder(inshape=observations_shape,scale_ob=1.0,outsize=outsize)
        self.pi_head = tu.NormLinear(nn.Linear(outsize,n_actions),scale=0.1)
        self.vi_head = tu.NormLinear(nn.Linear(outsize,1),scale=0.1)
    
    def forward(self,x):
        x = self.encoder(x)
        logits:T.Tensor = self.pi_head(x)
        probs = logits.softmax(dim=-1)
        v = self.vi_head(x)
        return probs , v

    def act(self, observation: T.Tensor) -> T.Tensor:
        prob,_ = self(observation)
        return prob
    def evaluate(self, observation: T.Tensor) -> T.Tensor:
        _,v = self(observation)
        return v
    def act_and_eval(self, observation: T.Tensor) -> tuple[T.Tensor, T.Tensor]:
        prob,v = self(observation)
        return prob,v
    
class ImpalaCritic(CriticNetwork):
    def __init__(self,observations_shape) -> None:
        super().__init__()
        outsize = 256
        self.encoder = ImpalaEncoder(inshape=observations_shape,scale_ob=1.0,outsize=outsize)
        self.pi_head = tu.NormLinear(nn.Linear(outsize,1),scale=0.1)
    def forward(self,x):
        x = self.encoder(x)
        x:T.Tensor = self.pi_head(x)
        return x
    def evaluate(self, observation: T.Tensor) -> T.Tensor:
        x = self(observation)
        return x


class RnnActorCriticNetwork(PytorchNetwork):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_initials(self,n:int)->tuple[T.Tensor,T.Tensor]:
        raise NotImplementedError()
    
    @abstractmethod
    def encode(self,x:T.Tensor)->tuple[T.Tensor]:
        raise NotImplementedError()

    @abstractmethod
    def single(self,x:T.Tensor,ht:T.Tensor,ct:T.Tensor)->tuple[T.Tensor,T.Tensor,T.Tensor,T.Tensor]:
        raise NotImplementedError()
    
    @abstractmethod
    def predict(self,x:T.Tensor,ht:T.Tensor,ct:T.Tensor)->tuple[T.Tensor,T.Tensor,T.Tensor,T.Tensor]:
        raise NotImplementedError()


class ImpalaRnnActorCriticNetwork(RnnActorCriticNetwork):
    def __init__(self,observation_shape:tuple,n_actions:int) -> None:
        super().__init__()
        self.lstmout = 256
        self.encoder = ImpalaEncoder(observation_shape,scale_ob=1.0,outsize=256)
        self.lstm = nn.LSTMCell(256,self.lstmout)
        self.pi_head = nn.Sequential(
            nn.Linear(self.lstmout,n_actions)
        )
        self.v_head = nn.Sequential(
            nn.Linear(self.lstmout,1)
        )

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.lstm.to(self.device)
        self.pi_head.to(self.device)
        self.v_head.to(self.device)


        self.ht0 = T.nn.Parameter(T.zeros(1,self.lstmout,dtype=T.float32,device=self.device),requires_grad=True)
        self.ct0 = T.nn.Parameter(T.zeros(1,self.lstmout,dtype=T.float32,device=self.device),requires_grad=True)
    
    def get_initials(self,n=1):
        ht = self.ht0.expand((n,self.lstmout))
        ct = self.ct0.expand((n,self.lstmout))
        return ht,ct

    def forward(self,obs:T.Tensor,htc:tuple[T.Tensor,T.Tensor]):
        x = self.encoder(obs)
        ht,ct = htc
        ht,ct = self.lstm(x,(ht,ct))
        v = self.v_head(ht)
        logits:T.Tensor = self.pi_head(ht)
        probs = logits.softmax(dim=-1)
        return probs,v,ht,ct
    
    def encode(self,x:T.Tensor):
        t,w =  x.shape[:-3]
        x = x.reshape(t*w,*x.shape[-3:])
        x = self.encoder(x)
        x = x.reshape(t,w,*x.shape[1:])
        return x
    
    def single(self,x:T.Tensor,ht:T.Tensor,ct:T.Tensor):
        ht,ct = self.lstm(x,(ht,ct))
        v = self.v_head(ht)
        logits:T.Tensor = self.pi_head(ht)
        probs = logits.softmax(dim=-1)
        return probs,v,ht,ct

    def stateless_forward(self,obs:T.Tensor):
        n = obs.size(1)
        ht,ct = self.get_initials(n)
        return self.predict(obs,(ht,ct))
    
    def predict(self,obs:T.Tensor,ht:T.Tensor,ct:T.Tensor)->tuple[T.Tensor,T.Tensor,T.Tensor,T.Tensor]:
        return self(obs,(ht,ct))

class SmallRnnActorCriticNetwork(PytorchNetwork):
    def __init__(self,observation_shape:tuple,n_actions:int) -> None:
        super().__init__()
        self.lstmout = 256
        self.encoder = nn.Sequential(
            # (?,4,84,84)
            nn.Conv2d(observation_shape[0], 32, 8, 4),
            # (?,32,20,20)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            # (?,64,9,9)
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            # (?,64,7,7)
            nn.ReLU(),
            nn.Flatten(),
        )
        self.lstm = nn.LSTMCell(64*7*7,self.lstmout)
        self.pi_head = nn.Sequential(
            nn.Linear(self.lstmout,n_actions),
        )
        self.v_head = nn.Sequential(
            nn.Linear(self.lstmout,1)
        )

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.lstm.to(self.device)
        self.pi_head.to(self.device)
        self.v_head.to(self.device)


        self.ht0 = T.nn.Parameter(T.zeros(1,self.lstmout,dtype=T.float32,device=self.device),requires_grad=True)
        self.ct0 = T.nn.Parameter(T.zeros(1,self.lstmout,dtype=T.float32,device=self.device),requires_grad=True)
    
    def get_initials(self,n=1):
        ht = self.ht0.expand((n,self.lstmout))
        ct = self.ct0.expand((n,self.lstmout))
        return ht,ct

    def forward(self,obs:T.Tensor,htc:tuple[T.Tensor,T.Tensor]):
        x = self.encoder(obs)
        ht,ct = htc
        ht,ct = self.lstm(x,(ht,ct))
        v = self.v_head(ht)
        logits:T.Tensor = self.pi_head(ht)
        probs = logits.softmax(dim=-1)
        return probs,v,ht,ct
    
    def encode(self,x:T.Tensor):
        t,w =  x.shape[:-3]
        x = x.reshape(t*w,*x.shape[-3:])
        x = self.encoder(x)
        x = x.reshape(t,w,*x.shape[1:])
        return x
    
    def single(self,x:T.Tensor,ht:T.Tensor,ct:T.Tensor):
        ht,ct = self.lstm(x,(ht,ct))
        v = self.v_head(ht)
        logits:T.Tensor = self.pi_head(ht)
        probs = logits.softmax(dim=-1)
        return probs,v,ht,ct

    def stateless_forward(self,obs:T.Tensor):
        n = obs.size(1)
        ht,ct = self.get_initials(n)
        return self.predict(obs,(ht,ct))
    
    def predict(self,obs:T.Tensor,ht:T.Tensor,ct:T.Tensor)->tuple[T.Tensor,T.Tensor,T.Tensor,T.Tensor]:
        return self(obs,(ht,ct))


class SmallDuelingQNetwork(StateActionNetwork):
    def __init__(self , observation_shape:tuple,n_actions:int) -> None:
        super().__init__()
    def evaluate(self, observation: T.Tensor)->T.Tensor:
        return self(observation)
    def forward(self,observation:T.Tensor):
        raise NotImplementedError()

class DuelingQNetwork(StateActionNetwork):
    def __init__(self,observation_shape:tuple,n_actions:int) -> None:
        super().__init__()
        self._shared = nn.Sequential(
            # (?,4,84,84)
            nn.Conv2d(observation_shape[0], 32, 8, 4),
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
            nn.ReLU()
            )
        self._advantages = nn.Linear(512,n_actions)
        self._value = nn.Linear(512,1)

    
    def evaluate(self, observation: T.Tensor)->T.Tensor:
        return self(observation)
    
    def forward(self,observation:T.Tensor):
        shared :T.Tensor = self._shared(observation)

        advs : T.Tensor = self._advantages(shared)
        advs = advs - advs.mean(dim=-1,keepdim=True)
        value = self._value(shared)
        qsa = advs + value
        return qsa