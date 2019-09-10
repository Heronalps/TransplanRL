import numpy as np


# Based on the histogram, working time on translations is inline with Exponential / Pareto Distribution
# This function synthesize distributions(i.e. Pareto, Gaussian)

def synthesize(dist="Gaussian", sample_size=1000, parameters={}):
    if dist == "Gaussian":
        sample = np.random.normal(parameters['mean'], parameters['std'], sample_size)
    elif dist == "Pareto":
        sample = (np.random.pareto(parameters['shape'], sample_size) + 1) * parameters['scale']
        sample = (np.random.pareto(0.85, 100) + 1) * 5
        # Truncate the long tail to avoid large neagtive reward
        sample = np.clip(sample, parameters['scale'], parameters['scale'] * 20)
    return sample

if __name__ == "__main__":
    pass