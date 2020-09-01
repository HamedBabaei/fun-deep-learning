
import numpy as np

def layer_size(X, Y):
    '''
       Returns layer sizes
    Arguments:
       X : input dataset of shape (input size, number of examples)
       Y : labels of shape(output size, number of examples)
    '''
    n_x = X.shape[0]
    n_h = 4   #default is set to 4
    n_y = Y.shape[0]
    return (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y):
    '''
       Initialize Parameters
    Argument:
       n_x: size of input layer 
       n_h: size of hidden layer
       n_y: size of output layer
    Returns:
       params(dict): {"W1":W1, "b1":b1, "W2":W2, "b2":b2}
    '''
    np.random.seed(2)

    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))

    params = {"W1":W1, "b1":b1, "W2":W2, "b2":b2}
    return params

