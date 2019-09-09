import numpy as np

from interface import Simulator, ActionSpace


# In the simplied version, two actions are in the space: [vote, collection]
# Simulator will set up a reward policy for agent to learn by Q-learning

class TransplanSimulator(Simulator):
    def __init__(self, num_translation_string=100, 
                       init_num_candidate=3, 
                       min_majority = 5,
                       sla = 3600 * 24, 
                       budget = 100 * 100,
                       tolerance_rate = 0,
                       cost_range = [0.01, 0.1]):
        
        # Set up parameters of MTurk
        self._num_translation_string = num_translation_string # The number of vote job on one translation issue
        self._init_num_candidates =  init_num_candidate
        self._min_majority = min_majority 
        self._num_worker = min_majority + (min_majority - 1) * (init_num_candidate - 1) # The number of workers to gain at least minimum majority
        self._sla = sla
        self._budget = budget
        self._tolerance_rate = tolerance_rate
        self._cost_range = cost_range
    
    

    def take_action(self, action):
        reward = 0

        if random.random() < self.slip:
            # Randomly select an action from space
            rand_int = random.randint(1, len(action))
            action = ActionSpace(rand_int)
        if action == ActionSpace.VOTE:
            pass
        if action == ActionSpace.COLLECTION:
            pass
    
        return self.state, reward

    def reset(self):
        self.state = 0
        return self.state