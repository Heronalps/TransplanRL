'''
Algorithm interface is the base class of RL learning algorithms
'''

import numpy as np

class Algorithm(object):

    """
        Parameters:
            simulator: instance of Simulator
            batch_size: number of actions taken in each iteration of Gradient Descent
    """

    def __init__(self, simulator, batch_size):
        self._simulator = simulator
        self._lr = 0.001
        self._df = 0.9
        self._batch_size = batch_size
        self._action_dimensions = simulator.get_action_dimensions()
        self._num_actions = simulator.get_num_actions()

    def train(self, states, actions, rewards, nextStates, terminals):
        """
            This function performs one training step for one batch of tuples. 
        """
        raise NotImplementedError()

    def get_best_action(self, state):
        """
            Get the best action for a pseudo-state
        """
        raise NotImplementedError()

    def get_qvalue(self, state):
        """
            Get the q value for one pseudo-state 
        """
        raise NotImplementedError()

    def set_learning_rate(self, lr):
        """
            This function sets learning rate if users want to finetune
        """
        self._lr = lr
    
    def set_discount_factor(self, df):
        """
            This function sets discount factor if users want to finetune
        """
        self._df = df
    
    def get_learning_rate(self):
        """
            Get the learning rate 
        """
        return self._lr

    def get_discount_factor(self):
        """
            Get the discount factor
        """
        return self._df
