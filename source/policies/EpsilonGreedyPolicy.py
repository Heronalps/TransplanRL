from ..interface import Policy

class EpsilonGreedyPolicy(Policy):
    """
        This policy acts greedily with probability '1-epsilon' and acts randomly otherwise.
    """

    def __init__(self, learning_algo, num_actions, random_state, epsilon):
        Policy.__init__(self, learning_algo, num_actions, random_state)
        self._epsilon = epsilon
    
    def action(self, state, mode=None, *args, **kwargs):
        if self.random_state.rand() < self._epsilon:
            action, value = self.random_action()
        else:
            action, value = self.best_action(state, mode, *args, **kwargs)

        return action, value
    
    def setEpsilon(self, e):
        self._epsilon = e

    def epsilon(self):
        return self._epsilon
    