import numpy as np


# Based on the histogram, working time on translations is inline with Exponential / Pareto Distribution
# This function synthesize distributions(i.e. Pareto, Gaussian)

def synthesize(dist="Gaussian", sample_size=1000, parameters={}, random_generator=None):

    if not random_generator:
        random_generator = np.random.RandomState(123456)
    if dist == "Gaussian":
        sample = random_generator.normal(parameters['mean'], parameters['std'], sample_size)
    elif dist == "Pareto":
        sample = (random_generator.pareto(parameters['shape'], sample_size) + 1) * parameters['scale']
        # Truncate the long tail to avoid large neagtive reward
        sample = np.clip(sample, parameters['scale'], parameters['scale'] * 20)
    return sample

"""
This function parses [constituency (0, 1), candidate (0, 1, 2), voting (0, 1), cost]
       [Constituency, Candidate, Voting]
0-9 => [0, 0, 0]
10-19 => [0, 0, 1]
20-29 => [0, 1, 0]
...
110-119 => [1, 2, 1]

"""
def parse_action(action):
    if action == 120:
        import pdb; pdb.set_trace();
    result = [0, 0, 0, 0]
    
    # Cost is the last digit of action number
    result[3] = action % 10
    action = action // 10
    if action == 0:
        pass
    elif action == 1:
        result[2] = 1
    elif action == 2:
        result[1] = 1
    elif action == 3:
        result[2] = 1
        result[1] = 1
    elif action == 4:
        result[1] = 2
    elif action == 5:
        result[2] = 1
        result[1] = 2
    elif action == 6:
        result[0] = 1
    elif action == 7:
        result[0] = 1
        result[2] = 1
    elif action == 8:
        result[1] = 1
        result[0] = 1
    elif action == 9:
        result[0] = 1
        result[1] = 1
        result[2] = 1
    elif action == 10:
        result[1] = 2
        result[0] = 1
    elif action == 11:
        result[2] = 1
        result[1] = 2
        result[0] = 1
    
    return result[0], result[1], result[2], result[3]


if __name__ == "__main__":
    pass