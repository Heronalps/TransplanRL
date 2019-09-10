import numpy as np
import pandas as pd
import math 

from interface import Simulator, ActionSpace
from helper.helpers import synthesize


# In the simplied version, two actions are in the space: [vote, collection]
# Simulator will set up a reward policy for agent to learn by Q-learning

class TransplanSimulator(Simulator):
    def __init__(self, rng, 
                       num_translation_string=100, 
                       init_num_candidate=3, 
                       min_majority = 5,
                       sla = 3600 * 24, # 24 hours in seconds
                       budget = 100 * 100, # 100 dollars in cents
                       tolerance_rate = 0,
                       cost_range = list(range(1, 11))):
        
        # Set up parameters of MTurk
        self.num_translation_string = num_translation_string # The number of vote job on one translation issue
        self.num_candidates =  init_num_candidate
        self.min_majority = min_majority 
        self.num_worker = min_majority + (min_majority - 1) * (init_num_candidate - 1) # The number of workers to gain at least minimum majority
        self.sla = sla
        self.budget = budget
        self.tolerance_rate = tolerance_rate
        self.cost_range = cost_range

        # Define the observation of Simulator
        # At each tiem step, the observation is made up of three elements: Response Time, Work Time, Success Rate
        self.last_punctual_observation = [0, 0, 0]
        self.random_state = rng

        # Fixed configurations from MTurk experiment

        self.std = 117 # Fixed number from EN-GB for now
        self.shape = 0.825 # Fixed number from EN-GB for now
        self.translation_time = 3600 # seconds
        self.translation_cost = 0.05 * self.num_worker
        self.work_time_lower_bound = 3 * self.num_candidates
        self.work_time_upper_bound = self.work_time_lower_bound * 10 # Assume 10 times lower bound = upper bound
        self.min_response_time = 74 # seconds

        self.episode_counter = 0
        """
            Constituency: Same/Different = [0.8, 1] 
            Candidates: Eliminate least voted / Same candidate / Additional Candidate = [1, 0.8, 1.2] 
            Voting: Single-out / Instant-runoff (Preferential) = [0.8, 1] 
            Cost: [$0.01, $0.1] 
        """
        self.constituency_rate_vector = [0.8, 1]
        self.candidates_rate_vector = [1, 0.8, 1.2]
        self.voting_rate_vector = [0.8, 1] 

    def reset(self, mode):
        pass

    def act(self, action):
        """
            Perform one time-step within the environment and updates the last_punctual_observation

            Parameters:
            -----------
            action : vector
                [Constituency, Candidate, Voting, Cost]
                
                Constituency: Same/Different = [0.8, 1] 
                Candidates: Eliminate least voted / Same candidate / Additional Candidate = [1, 0.8, 1.2] 
                Voting: Single-out / Instant-runoff (Preferential) = [0.8, 1] 
                Cost: [$0.01, $0.1] 
                
            Returns:
            --------
            reward: float
            isTerminal: bool
        """
        # All jobs are finished
        if self.num_translation_string <= 0:
            return 0, True
        
        # No SLA or budget left
        if self.sla <= 0 or self.budget <= 0:
            return 0, True

        constituency = action[0]
        candidate = action[1]
        voting = action[2]
        cost = action[3]

        response_time_sample = synthesize(dist="Gaussian", 
                                          sample_size=self.num_translation_string,
                                          parameters={"mean": 1 / self.cost_range[cost], "std": self.std})
        
        # For different constituency, response time increase 20%
        if constituency == 1:
            response_time_sample = response_time_sample * 1.2
        

        # For work time, lower bound is set at 3 seconds * num_candidates
        work_time_sample = synthesize(dist="Pareto",
                                      sample_size=self.num_translation_string,
                                      parameters={"shape": self.shape, "scale": self.work_time_lower_bound})
        
        success_rate = 0
        if self.episode_counter == 0:
            success_rate = self.random_state.uniform(0.8, 1)
        else:
            # Same constituency + Same candidate => success rate = 0
            if constituency == 1 and candidate == 1:
                success_rate = 0
            else:
                success_rate = self.constituency_rate_vector[constituency] * \
                               self.candidates_rate_vector[candidate] * \
                               self.voting_rate_vector[voting]
        

        df = pd.DataFrame(data={
                            "response_time": response_time_sample,
                            "work_time": work_time_sample,
                            "success_rate": pd.Series([success_rate] * self.num_translation_string)
                            })

        # import pdb; pdb.set_trace();
        # Update last punctual obseration
        self.last_punctual_observation = df
        
        min_response_time = self.min_response_time
        work_time_lower_bound = self.work_time_lower_bound
        work_time_upper_bound = self.work_time_upper_bound

        # Cumulative Positive Rewards
        df['reward'] = df.apply(lambda row: success_rate * 
                                            ((1 / max(row['response_time'], min_response_time)) - 
                                            (row['work_time'] - work_time_lower_bound) * 
                                            (row['work_time'] - work_time_upper_bound)), axis=1)
        
        # Average out reward to number of translations
        reward = df['reward'].sum() / self.num_translation_string
        
        df['total_time'] = df.apply(lambda row: row['response_time'] + row['work_time'], axis=1)

        # Calculate time and money spent
        time = df['total_time'].max()
        money = self.cost_range[cost] * self.num_translation_string

        # In case of additional translation
        time += self.translation_time
        money += self.translation_cost

        # Deduct from SLA and Budget
        self.sla -= time
        self.budget -= money

        # Deduct successful translation from work order
        self.num_translation_string -= math.ceil(self.num_translation_string * success_rate)

        if self.num_translation_string <= 0:
            return reward, True
        else:
            return reward, False

    def get_action_dimension(self):
        return [(1, 3)]

    def get_num_actions(self):
        return len(self.constituency_rate_vector) * \
            len(self.candidates_rate_vector) * \
            len(self.voting_rate_vector)

    def inTerminalState(self):
        return self.num_translation_string <= 0

    def observe(self):
        return self.last_punctual_observation
    
    def get_constraints(self):
        return [self.sla, self.budget]

if __name__ == "__main__":
    rng = np.random.RandomState(123456)
    mySimulator = TransplanSimulator(rng)

    print (mySimulator.act([1, 1, 1, 9]))
    print (mySimulator.observe())
    