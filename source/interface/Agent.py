"""
Agent is the interface for learning agent. 
"""

import numpy as np

class Agent(object):

    '''
        Parameters:
            constraints: The dict of various constraints this agent has.
    '''
    def __init__(self, constraints=None):
        self._constraints = constraints
        self._reward = 0

