# This script runs the training process of TransplanRL

import numpy as np

from NeuralAgent import NeuralAgent
from algorithms.q_net_keras import QNetwork
from TransplanSimulator import TransplanSimulator
import experiment.base_controllers as bc

rng = np.random.RandomState(123456)

def run():
    simulator = TransplanSimulator(rng)
    q_network = QNetwork(environment=simulator, random_state=rng)
    agent = NeuralAgent(environment=simulator, learning_algo=q_network, random_state=rng)
    agent.attach(bc.VerboseController())
    agent.attach(bc.TrainerController())
    agent.attach(bc.InterleavedTestEpochController(
        epoch_length=5,
        controllers_to_disable=[0, 1]
    ))
    agent.run(n_epochs=10, epoch_length=10)


if __name__ == "__main__":
    run()