import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete


class GameEnv(gym.Env):
    """ A gym environment for a simplistic game world 
        in which the enironment dynamics (i.e state transition and reward)
        is known.

    Args:
    
        transitions (list): list of transition probability matrices (np.array)
        rewards (list): state-action rewards
    """

    def __init__(self, transitions, rewards):
        self.transitions = transitions
        self.rewards = rewards
        
        self.observation_space = Discrete(self.transitions.shape[1])
        self.action_space = Discrete(self.rewards.shape[1])

        self.current_state = None
        self.total_reward = None
        self.terminal_state = None

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, stop, info).

        Args:
            action: an action provided by the environment

        Returns:
            obs (object): agent's observation of the current environment (in our case the new state)
            reward (float) : amount of reward returned after previous action
            terminated (bool): whether the episode has ended
            truncated (bool): whether the episode was truncated
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        
        length = range(self.transitions.shape[1])
        # Get the tranisition probabiliteies of the current state given an action
        probability = self.transitions[action,self.current_state,: ]
        next_state = np.random.choice(length, p=probability)
        # Get the reward of the current state given an action
        reward = self.rewards[self.current_state, action]
        if next_state == self.terminal_state:
            done = True
        else:
            done = False
        self.current_state = next_state
        obs = next_state
        return obs, reward, done

    def reset(self, seed=None, options=None):
        """Resets the state of the environment and returns an initial observation.
        Returns: 
            observation (object): the initial observation of the space (in our case the initial state)
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        info = {}
        # implement the environment reset functionality here...
        self.current_state = 0 # state s0
        obs = self.current_state
        return obs, info

    def _render(self, mode='human', close=False):
        return NotImplemented
