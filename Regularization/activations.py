import numpy as np

def sigmoid(Z):
    '''
      Compute sigmoid of Z
    Arguments:
        Z: Output of the linear layer
    Return: 
        A: sigmoid(Z)
        cache: Z
    '''
    A = 1/(1 + np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    """
        Implement the RELU function.
    Arguments:
        Z : Output of the linear layer
    Returns:
        A: relu(z)
        cache: Z
    """
    A = np.maximum(0,Z)
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    """
        Implement the backward propagation for a single RELU unit.
    Arguments:
        dA: derevite of A post-activation gradient, of any shape
        cache: Z
    Returns:
        dZ: backward value
    """
    Z = cache
    dZ = np.array(dA, copy=True) 
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_backward(dA, cache):
    """
        Implement the backward propagation for a single SIGMOID unit.
    Arguments:
        dA: derevite of A post-activation gradient, of any shape
        cache: Z
    Returns:
        dZ: backward value
    """
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

