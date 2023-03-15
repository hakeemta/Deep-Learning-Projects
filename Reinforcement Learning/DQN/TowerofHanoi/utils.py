import numpy as np
from ray.rllib.utils.numpy import fc,relu




def softplus(x):
    return np.log(1 + np.exp(x))


def hard_sigmoid(x):
    return np.minimum(1, np.maximum(0, (0.2 * x) + 0.5))

def encoder(input, weights):
    activations = [softplus, hard_sigmoid, relu, softplus, softplus]
    dense = input
        
    for i in range(0, len(weights), 2):
        dense = fc(dense, weights[i], weights[i+1])
        activation = activations[i//2](dense)
        dense = activation
        
    return dense


