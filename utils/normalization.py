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

def normalize_image(X):
     '''
        Image (b=count, c=width, d=height, a=channels(rgb = 3)) where 
        Flatted image Xi is (b*c*d, a)

        Flatted image Xi normalization by divide 255
     Argument: 
          X: a numpy matrix of shape (b*c*d, a)
     return:
          norm: normalized image
     '''
     norm = X/255.
     return norm