import numpy as np
import __init__
from utils.activation import sigmoid

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

def forward_propagation(X, parameters):
   '''
      Forward Propagation
   Arguments:
      X : input data with size of (n_x, m)
      parameters: dictionary contains {"W1":W1, "b1":b1, "W2":W2, "b2":b2}
   Returns:
      A2 : output of second activation
      cache: a dictoinary of Z1, A1, Z2, A2 for backward calculation
   '''
   W1 = parameters['W1']
   b1 = parameters['b1']
   W2 = parameters['W2']
   b2 = parameters['b2']

   Z1 = np.dot(W1, X) + b1
   A1 = np.tanh(Z1)
   Z2 = np.dot(W2, A1) + b2
   A2 = sigmoid(Z2)

   cache = {"Z1":Z1, "A1":A1, "Z2":Z2, "A2":A2}

   return A2, cache

def compute_cost(A2, Y):
   '''
      computes the cross-entropy cost
   Arguments:
      A2: the sigmoid output of the second activation
      Y : true label vector of shape(1, number of example)
      parameters: dictionary contains {"W1":W1, "b1":b1, "W2":W2, "b2":b2}
   Returns:
      cost
   '''
   m = Y.shape[1]
   logprops = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
   cost = - (1/m) * np.sum(logprops)
   cost = float(np.squeeze(cost))
   return cost

def backward_propagation(parameters, cache, X, Y):
   '''
      Backward Propogation
   Arguments;
      parameters: dictionary contains {"W1":W1, "b1":b1, "W2":W2, "b2":b2}
      cache: dictionary contains {"Z1":Z1, "A1":A1, "Z2":Z2, "A2":A2} from forward propogation
      X : input data of shape (2, number of examples) 2 is a feature number
      Y : true labels vector of shape (1, number of examples)
   Returns:
      grads: dictionary contains {"dW1": dW1, "db1":db1, "dW2": dW2, "db2":db2}
   '''
   m = X.shape[1]

   W1 = parameters['W1']
   W2 = parameters['W2']
   A1 = cache['A1']
   A2 = cache['A2']

   dZ2 = A2 - Y
   dW2 = (1/m)*np.dot(dZ2, A1.T)
   db2 = (1/m)*np.sum(dZ2, axis=1, keepdims=True)
   dZ1 = np.dot(W2.T, dZ2)*(1- np.power(A1, 2))
   dW1 = (1/m)*np.dot(dZ1, X.T)
   db1 = (1/m)*np.sum(dZ1, axis=1, keepdims=True)
   
   grads = {"dW1": dW1, "db1":db1, "dW2": dW2, "db2":db2}
   return grads

def update_parameters(parameters, grads, learning_rate = 1.2):
   pass

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False, learning_rate = 1.2):
   pass

def predict(parameters, X):
   pass

