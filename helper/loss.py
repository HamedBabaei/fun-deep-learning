import numpy as np


def L1(yhat, y):
    '''
      Compute L1 loss
    Arguments:
      yhat: vector of size m (predicted labels)
      y: vector of size m (true labels)
    Returns:
      loss: the value of the L1 loss function
    '''
    loss = np.sum(abs(yhat - y))
    return loss


def L2(yhat, y):
    '''
      Compute L2 loss
    Arguments:
      yhat: vector of size m (predicted labels)
      y: vector of size m (true labels)
    Returns:
      loss: the value of the L2 loss function
    '''
    loss = np.sum(np.dot(yhat - y, yhat - y))
    return loss


