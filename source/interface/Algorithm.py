'''
Algorithm interface is the base class of RL learning algorithms
'''

import numpy as np
from Simulator import Simulator 

class Algorithm(object):

    '''
        Parameters:
            simulator: instance of Simulator
            batch_size: number of actions taken in each iteration of Gradient Descent
    '''

    def __init__(self, simulator, batch_size):
        self._simulator = simulator
        self._lr = 0.001
        self._batch_size = batch_size
        self._action_dimensions = simulator.get_action_dimensions()
        self._num_actions = simulator.get_num_actions()
        
