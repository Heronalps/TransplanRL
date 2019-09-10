'''
Simulator is the interface of the training environment.
'''

import numpy as np

class Simulator(object):
    """
        The base class of Simulator Environment
    """
    def reset (self, mode):
        """
            Reset the simulator to a spcific mode when beginning every new episode
        """
        raise NotImplementedError()

    def act (self, action):
        """
            Apply a certain action to the simulator
            -----------
            Parameters:
            -----------
                action: vector
            -----------      
            Returns:
                isSuccess: bool
        """
        raise NotImplementedError()
    
    def get_action_dimension(self):
        """
            Gets a list of action tuple. The history_action_size represents the size of observations 
            taken into account at this step.
            -----------
            Returns:
                dimensions: [(history_action_size, dimension...),...] 
        """
        raise NotImplementedError()

    def get_num_actions(self):
        """
            Gets the number of actions in the discrete action space
            -----------
            Returns:
                num: int
        """
        raise NotImplementedError()

    def get_state_dimension(self):
        """
            Get the state dimension tuple of 
            -----------
            Returns:
                dimensions: (dimensions) 
        """
        
        raise NotImplementedError()

    def observe(self):
        """
            Observe the current state of the simulator
            -----------
            Returns:
                state: matrix of defined dimension
        """

        raise NotImplementedError()

    def isTerminated(self):
        """
            Tells if the simulator is in predefined terminal state
            -----------
            Returns:
                isTerminated: bool
        """
        raise NotImplementedError()
