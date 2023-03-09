import gymnasium
import numpy as np
import itertools
import tensorflow as tf



class TowerOfHanoiEnv(gymnasium.Env):
    def __init__(self,config) -> None:
        super(TowerOfHanoiEnv).__init__()
        self.num_towers=3
        self.num_disks = config["num_disks"]
        self.action_space = gymnasium.spaces.Discrete(6)
        self.observation_space = gymnasium.spaces.MultiDiscrete([[self.num_disks+1]*self.num_disks]*self.num_towers)
        self.position =list(itertools.permutations(list(range(self.num_towers)), 2))
    
    def reset(self,seed=None, options=None):
        """Resets the state of the environment and returns an initial observation.
        Returns: 
            observation (object): the initial observation of the space (in our case the initial state)
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        self.state = np.vstack((np.arange(self.num_disks, dtype=int),    # Source Tower
                                np.full(self.num_disks, 0),         # Helper Tower
                                np.full(self.num_disks, 0)))        #Destination Tower
        # Reverse the order of the disks in source tower
        self.state[0, -self.num_disks:] = self.state[0, :self.num_disks][::-1]
        self.state[0, :] = self.state[0] +1
        return self.state, {}
    
    def step(self, action):
        """Run one time-step of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
    
        Args:
            action: an action provided by the environment
    
        Returns:
            obs (object): agent's observation of the current environment
            reward (int) : amount of reward returned after previous action
            terminated (bool): whether the episode has ended
            truncated (bool): whether the episode was truncated
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        truncated = None
        action_idx = self.position[action]
        from_tower, to_tower = action_idx[0], action_idx[1]
        preprocessed_state = self.preprocess()
        # Check the validity of the move
        condition =  min(preprocessed_state[to_tower]) != np.inf  and min(preprocessed_state[from_tower]) > min(preprocessed_state[to_tower]) 
    
        if condition or min(preprocessed_state[from_tower]) == min(preprocessed_state[to_tower]) :
            # Invalid move
            return self.state, float(-0.5), False,False, {}
    
        # Valid move
        disk = min(preprocessed_state[from_tower])
        disk_idx = np.argmin(preprocessed_state[from_tower])
        to_idx = np.argmax(preprocessed_state[to_tower])
        # Move disk to a new position
        self.state[to_tower][to_idx] = disk
        # np.inf is use to denote that a disk at a position is empty
        # ToDo : np.inf is not an appropriate choice in this case
        self.state[from_tower][disk_idx] = 0
    
        # Check if we are at the terminal state
        # A reward of -1 is given for every valid move
        if np.all(self.state[2] != 0):
            return  self.state,float(-1), True,True,{}
    
        return self.state, float(-1), False, False,{}
    
    def preprocess(self):
        return np.where(self.state ==0, np.inf, self.state)
    



class FlattenEnv(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gymnasium.spaces.Box(shape=(30,), low=-np.inf, high=np.inf)
            
    # Override `observation` to custom process the original observation
    # coming from the env.
    def observation(self, observation):
        return observation.flatten()




class EmbeddingEnv(gymnasium.ObservationWrapper):
# Override `observation` to custom process the original observation
# coming from the env.
    def __init__(self, env):
         super().__init__(env)
         self.observation_space = gymnasium.spaces.Box(shape=(16,), low=-np.inf, high=np.inf)
    def observation(self, observation):
        model = tf.keras.models.load_model('autoencoder')
        #model = AutoEncoder()
        #model.load_weights('./checkpoints/my_checkpoint')
        return model.encoder(observation)