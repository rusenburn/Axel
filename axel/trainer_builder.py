from ppo.ppo_builder import PpoBuilder
from ppg.ppg_builder import PpgBuilder
from pop3d.pop3d_builder import Pop3dBuilder

class TrainerBuilder():
    def __init__(self) -> None:
        ...

    def use_ppo(self)->PpoBuilder:
        return PpoBuilder()
    
    def use_ppg(self):
        return PpgBuilder()
    
    def use_pop3d(self):
        return Pop3dBuilder()