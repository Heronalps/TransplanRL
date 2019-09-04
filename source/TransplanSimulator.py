import random

from Simulator import Simulator 
from ActionSpace import ActionSpace


# In the simplied version, two actions are in the space: [vote, collection]
# Simulator will set up a reward policy for agent to learn by Q-learning

class TransplanSimulator(Simulator):
    def __init__(self, num_votes=3, error_rate=0.1, num_candidate=3, cost_per_vote=0.01, 
                       slip=0.1):
        
        # Set up parameters of MTurk 
        self.num_votes = num_votes # The number of vote job on one translation issue
        self.error_rate = error_rate # The percentage of issues that do not have a valid winning candidate(NOTA/Non-significant Vote)
        self.num_candidate = num_candidate # The number of candidate for each issue
        self.cost_per_vote = cost_per_vote # The monetary cost per vote 
    
        # Set up config
        self.slip = slip # The probability of flipping a action
        self.state = 0
        self.action = ActionSpace(1) # 1: Vote; 2: Collection
    

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