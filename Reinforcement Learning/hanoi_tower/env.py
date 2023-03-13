from collections import deque
from enum import Enum
import numpy as np
import gymnasium
from gymnasium import spaces


class Towers(Enum):
    SOURCE = 0
    HELPER = 1
    DESTINATION = 2

class HanoiTower(gymnasium.Env):
    def __init__(self, config) -> None:
        super().__init__()
        self.n_towers = 3
        self.n_disks = config['n_disks']
        self.max_n_steps = config['max_n_steps']

        self.action_space = spaces.MultiDiscrete([3, 3])
        self.observation_space = spaces.Box(low=0, high=self.n_disks, 
                                            shape=(self.n_towers, self.n_disks), 
                                            dtype=np.int32)
        self._create_towers()

    def _create_towers(self):
        disk_radi = np.arange(self.n_disks) + 1
        self._towers = [deque(disk_radi, maxlen=self.n_disks)]
        self._towers += [deque(maxlen=self.n_disks)
                         for _ in range(self.n_towers-1) ]

    def _process_tower(self, _tower):
        _tower = list(_tower)
        size = self.n_disks - len(_tower)
        return np.pad(_tower, pad_width=(size, 0))
    
    def _extract_state(self):
        obs = [self._process_tower(t) 
               for t in self._towers]
        return np.array(obs, dtype=np.int32)

    def reset(self, seed=None, options=None):
        self._steps_count = 0
        self._create_towers()
        return self._extract_state(), {}
    
    def _validate_action(self, from_idx, to_idx):
        if from_idx == to_idx:
            return False
        
        if len(self._towers[from_idx]) < 1:
            return False
        
        if len(self._towers[to_idx]) < 1:
            return True

        if self._towers[from_idx][0] > self._towers[to_idx][0]:
            return False
        
        return True
    
    def step(self, action):
        from_idx, to_idx = action

        self._steps_count += 1
        terminated = (self._steps_count >= self.max_n_steps)
        if not self._validate_action(from_idx, to_idx):
            msg = 'Invalid action'
            return self._extract_state(), -1.0, False, terminated, {'log': msg}
        
        from_tower = self._towers[from_idx]
        to_tower = self._towers[to_idx]

        # Move
        disk = from_tower.popleft()
        to_tower.appendleft(disk)
        
        done = len(self._towers[Towers.DESTINATION.value]) == self.n_disks        
        return self._extract_state(), -1.0, done, terminated, {}
    

class PreprocessorEnv(gymnasium.ObservationWrapper):
    def observation(self, obs):
        obs = obs / obs.max()
        return obs

def create_env(config):
    _env = HanoiTower(config)
    _env = PreprocessorEnv(_env)
    return _env
