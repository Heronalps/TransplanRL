"""
    Policy is the interface for RL Policy. 
"""

import numpy as np

class Policy(object):
    """
        A policy takes observations as input, and outputs an action.
        Parameters:
        -----------
        learning_algo: object from class Algorithm
        actions: (int) Number of actions in the action space provided by Simulator.get_num_action()
        random_state: numpy random generator by certain distribution
    """

    def __init__(self, learning_algo, actions, random_state):
        self.learning_algo = learning_algo
        self.actions = actions
        self.random_state = random_state

    def best_action(self, state, mode=None, *args, **kwargs):
        """
            This function returns the best action for the given state.
        """
        action, value = self.learning_algo.get_best_action(state, mode, *args, **kwargs)
        return action, value

    def random_action(self):
        """
            This function returns a random action to explore unknown area
        """
        action = self.random_state.randint(0, self.actions)
        value = 0
        return action, value

    def action(self, state):
        """
            This function should be called by agent, given a state, and should return a valid 
            action to the simulator provided to the constructor
        """
        raise NotImplementedError()
    
