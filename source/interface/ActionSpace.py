from enum import Enum, auto

class ActionSpace(Enum):
    '''
        Parameters:
            actions : list of the name of actions
    '''
    def __init__(self, actions):
        for action in actions:
            action = auto()