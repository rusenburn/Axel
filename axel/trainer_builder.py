from .ppo.ppo_builder import PpoBuilder
from .ppg.ppg_builder import PpgBuilder
from .pop3d.pop3d_builder import Pop3dBuilder

class TrainerBuilder():
    def __init__(self) -> None:
        ...

    @staticmethod
    def use_ppo()->PpoBuilder:
        builder =  PpoBuilder()
        return builder
    
    @staticmethod
    def use_ppg():
        return PpgBuilder()
    
    @staticmethod
    def use_pop3d():
        return Pop3dBuilder()