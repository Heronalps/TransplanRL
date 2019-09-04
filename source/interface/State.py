"""
    The State interface is for the state of simulator.
"""

import numpy as np

class State(object):

    def __init__(self, attributes):
        self._attributes = attributes