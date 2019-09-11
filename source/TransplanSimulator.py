import numpy as np
import pandas as pd
import math 

from interface import Simulator, ActionSpace
from helper.helpers import synthesize, parse_action


# In the simplied version, two actions are in the space: [vote, collection]
# Simulator will set up a reward policy for agent to learn by Q-learning

class TransplanSimulator(Simulator):
    def __init__(self, rng, 
                       num_translation_string=100, 
                       init_num_candidates=3, 
                       min_majority = 5,
                       sla = 3600 * 2, # 2 hours in seconds
                       budget = 100 * 7, # 7 dollars in cents
                       tolerance_rate = 1e-3,
                       cost_range = list(range(1, 11))):
        
        # Set up parameters of MTurk
        self.origin_num_translation_string = num_translation_string # The number of vote job on one translation issue
        self.num_translation_string = num_translation_string # The number of vote job on one translation issue
        
        self.num_candidates = init_num_candidates
        self.min_majority = min_majority 
        self.num_worker = min_majority + (min_majority - 1) * (init_num_candidates - 1) # The number of workers to gain at least minimum majority
        
        self.origin_sla = sla
        self.sla = sla
        self.origin_budget = budget
        self.budget = budget
        self.tolerance_rate = tolerance_rate
        self.cost_range = cost_range
        self.cumulative_success_rate = 0
        self.random_state = rng

        # Define the observation of Simulator
        # At each time step, the observation is [cumulative success rate, action]
        self.last_punctual_observation = [0, 0]
        

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
        if mode == -1:
            self.num_translation_string = self.origin_num_translation_string 
        else:
            self.num_translation_string = 100

        self.sla = self.origin_sla
        self.budget = self.origin_budget
        self.num_candidates = 3
        self.cumulative_success_rate = 0
        self.episode_counter = 0

        return [1*[0], 0]

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
            
        """

        constituency, candidate, voting, cost = parse_action(action)

        # Update num of candidate
        if candidate == 0:
            if self.num_candidates <= 0:
                candidate = 1
            else:
                self.num_candidates -= 1

        if candidate == 2:
            self.num_candidates += 1 

        response_time_sample = synthesize(dist="Gaussian", 
                                          sample_size=self.num_translation_string,
                                          parameters={"mean": 1 / self.cost_range[cost], "std": self.std},
                                          random_generator=self.random_state)
        
        # For different constituency, response time increase 20%
        if constituency == 1:
            response_time_sample = response_time_sample * 1.2
        

        # For work time, lower bound is set at 3 seconds * num_candidates
        work_time_sample = synthesize(dist="Pareto",
                                      sample_size=self.num_translation_string,
                                      parameters={"shape": self.shape, "scale": self.work_time_lower_bound},
                                      random_generator=self.random_state)
        
        success_rate = 0
        if self.episode_counter == 0:
            success_rate = self.random_state.uniform(0.8, 1)
        else:
            # Same constituency + Same candidate => success rate = 0
            # Assign a large negative number as a punishment
            if constituency == 1 and candidate == 1:
                return -9999
            else:
                success_rate = self.constituency_rate_vector[constituency] * \
                               self.candidates_rate_vector[candidate] * \
                               self.voting_rate_vector[voting]
        
        print ("success_rate : ", success_rate)

        df = pd.DataFrame(data={"response_time": response_time_sample, "work_time": work_time_sample})

        # import pdb; pdb.set_trace();
        
        min_response_time = self.min_response_time
        work_time_lower_bound = self.work_time_lower_bound
        work_time_upper_bound = self.work_time_upper_bound

        # Cumulative Rewards - Quadratic function makes the rewards mostly likely negative 
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
        if candidate == 2:
            time += self.translation_time
            money += self.translation_cost

        # Deduct from SLA and Budget
        self.sla -= time
        self.budget -= money

        # Deduct successful translation from work order
        self.num_translation_string -= math.ceil(self.num_translation_string * success_rate)
        
        # Increase cumulative success rate
        self.cumulative_success_rate += (1 - self.cumulative_success_rate) * success_rate
        print ("Cumulative Success Rate : ", self.cumulative_success_rate)
        print ("In terminated state : ", self.inTerminalState())

        # Update episode_counter
        self.episode_counter += 1

        # Update last punctual observation
        self.last_punctual_observation[0] = self.cumulative_success_rate
        self.last_punctual_observation[1] = action

        return reward

    def get_action_dimension(self):
        return [(1,), (1,)]

    def get_num_actions(self):
        return 120

    def inTerminalState(self):
        isTerminated = (1 - self.cumulative_success_rate) <= self.tolerance_rate or \
                       self.num_translation_string <= 0 or \
                       self.sla <= 0 or \
                       self.budget <= 0
        return isTerminated

    def observe(self):
        return self.last_punctual_observation
    
    def get_constraints(self):
        return [self.sla, self.budget]

    def get_episode_counter(self):
        return self.episode_counter

    def summarizePerformance(self, test_data_set, *args, **kwargs):
        print ("Test Episode action sequence")
        print ("Observations : ", test_data_set.observations())
        print ("Actions : ", test_data_set.actions())
        print ("Rewards : ", test_data_set.rewards())
        print ("Terminals : ", test_data_set.terminals())
        print ("Contraints : ", self.get_constraints())
        print ("Current Observation : ", self.observe())
        print ("Episode Counter : ", self.get_episode_counter())
        print ("==========================")


if __name__ == "__main__":
    rng = np.random.RandomState(1)
    mySimulator = TransplanSimulator(rng)

    print (mySimulator.act(110))
    print (mySimulator.get_constraints())
    print (mySimulator.observe())
    