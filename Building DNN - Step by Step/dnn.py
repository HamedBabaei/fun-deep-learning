import numpy as np
import activations

def initialize_parameters(n_x, n_h, n_y):
    '''
       Initialize Parameters
    Arguments:
       n_x: size of the input layer
       n_h: size of the hidden layer
       n_y: size of the output layer
    Returns:
        parameters: a dictionary contains {"W1":W1, "b1":b1, "W2":W2, "b2":b2}
    '''
    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y,1))
    parameters = {"W1":W1, "b1":b1, "W2":W2, "b2":b2}
    return parameters

def initialize_parameters_deep(layer_dims):
    '''
        Initialize Deep Neural Network Parameters
    Arguments:
        layer_dims: list of units for each layers
    Returns:
        parameters: a dictionary containing parameters "W1", "b1",......"WL","bL"
    '''
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L): 
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
    return parameters

def linear_forward(A, W, b):
    '''
        Implementation of the Linear part of a layer's forward propagation
    Arguments:
        A: activations from previous layer
        W: weight matrix
        b: bias vector
    Returns:
        Z: the input of the activation function
        cache: a tuple to hold A, W, b
    '''
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    '''
        Implement the forward propagation for the LINEAR->ACTIVATION layer
    Arguments:
        A_prev: activations from previous layer
        W: weights matrix
        b: bias vector
        activation(string): activation that will be in use
    Returns:
        A: activation values
        cache (linear_cache(A, W, b), activation_cache=Z): a tuple to hold linar_cache, and activation_cache
    '''
    acts = {"sigmoid":activations.sigmoid, "relu":activations.relu}
    Z, linear_cache = linear_forward(A_prev, W, b)
    A, activation_cache = acts[activation](Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters):
    '''
        Implementation of forward propagation for [LINEAR->RELU]*(L-1) -> LINEAR->SIGMOID
    Arguments:
        X: a data which is a numpy aray with shape of (input size, number of examples)
        parameters: output of initialization_parameters_deep()
    Returns:
        AL: post activation value
        caches: list of catesh containing every cache of linear_activation_forward()
                (L-1 of them, indexed from 0 to L - 1)
    '''
    caches = []
    A = X
    L = len(parameters)//2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, 
                                             parameters["W"+str(l)], 
                                             parameters["b"+str(l)], 
                                             activation="relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A,
                                          parameters["W"+str(l+1)], 
                                          parameters["b"+str(l+1)], 
                                          activation="sigmoid")
    caches.append(cache)
    return AL, caches

def compute_cost(AL, Y):
    '''
        Implement the cost function
    Argument:
        AL: probability vector corresponding to your label predictions in shape of (1, number of examples)
        Y: truth label
    Returns:
        cost: cross entropy cost
    '''
    m = Y.shape[1]
    cost = (-1/m)*np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL))
    cost = np.squeeze(cost)
    return cost




