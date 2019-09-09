"""
    DQN is the class for Deep Q-learning Network using Keras
"""

import numpy as np
from keras.models import Model
from keras.layers import Input, Layer, Dense, Flatten, concatenate, Activation, Conv2D,\
    MaxPooling2D, Reshape, Permute

class DQN():
    """
        Deep Q-learning Network using Keras

        Parameters
        ----------
        batch_size: (int) Number of tuples for each gredient descent iteration
        action_dimension: 
        num_actions:
        random_state: numpy random number generator. Set the random seed.
        action_as_input: (bool) Whether the action is input or as output
    """
    def __init__(self, batch_size, action_dimension, num_actions, random_state, action_as_input=False):
        self._batch_size = batch_size
        self._action_dimension = action_dimension
        self._num_actions = num_actions
        self._random_state = random_state
        self._action_as_input = action_as_input
    
    def _buildDQN(self):
        """
            Build DQN consistent with each type of input
        """
        layers = []
        outputs_conv = []
        inputs = []

        for index, dim in enumerate(self._action_dimension):
            # observation[i] is a vector. 
            # In TransplanRL, observation is (ResponseTime, WorkTime, Success Rate)
            if len(dim) == 2:
                if dim[0] > 3:
                    input = Input(shape=(dim[0], dim[1]))
                    inputs.append(input)
                    reshaped = Reshape((dim[0], dim[1], 1), input_shape=(dim[0], dim[1]))(input)
                    x = Conv2D(16, (2, 1), activation='relu', padding='valid')(reshaped) # Conv2D on history
                    x = Conv2D(16, (2, 2), activation='relu', padding='valid')(x) # Conv2D on history & features

                    out = Flatten()(x)
                else:
                    input = Input(shape=(dim[0], dim[1]))
                    inputs.append(input)
                    out = Flatten()(input)

            outputs_conv.append(out)
        
        if len(outputs_conv)>1:
            x = concatenate(outputs_conv)
        else:
            x = outputs_conv[0]
        
        # Stack fully-connected network on the top

        x = Dense(50, activation='relu')(x)
        x = Dense(50, activation='relu')(x)

        if (self._action_as_input==False):
            if (isinstance(self._n_actions, int)):
                out = Dense(self._n_actions)(x)
            else:
                out = Dense(len(self._n_actions))(x)
        else:
            out = Dense(1)(x)

        model = Model(inputs=inputs, outputs=out)
        layers = model.layers

        # Grab all the parameters
        params = [ param for layer in layers for param in layer.traninable_weights ]

        return model, params

if __name__ == '__main__':
    pass