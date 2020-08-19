import numpy as np


def normalize_rows(x):
    '''
      Implement a function tht normalizes each row of the matrix x
    Argument: 
         x: a numpy matrix of shape(n, m)
    return:
         x: the normalized by row numpy matrix
    '''
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x/x_norm
    return x

