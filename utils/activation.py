import numpy as np

def sigmoid(x):
    '''
      Compute sigmoid of X
    Arguments:
        x: A scaler or numpy array of any size
    Return: 
        s: sigmoid(x)
    '''
    s = 1/(1 + np.exp(-x))
    return s


def sigmoid_derivative(x):
    '''
      compute the gradient of the sigmoid 
    Arguments:
          x : A scaler or numpy array
    Return: 
          ds: computed gradient
    '''
    s = sigmoid(x)
    ds = s*(1-s)
    return ds


def softmax(x):
    '''
      calculate the soft max for each row of the input x
    Argument:
      x: a numpy matrix of shape(m,n)
    
    Returns:
      s: a numpy matrix equal to the softmax of x
    '''
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True) #axis=1 for rows
    s = x_exp/x_sum
    return s
