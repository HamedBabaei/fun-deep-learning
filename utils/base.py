import numpy as np
import matplotlib.pyplot as plt

def initialize_with_zeros(dim):
    '''
       create a vector of zeros of shape (dim, 1) for w and intialize b to 0
    arguments:
       dim(int): size of w vector
    return:
       w(numpy): zero matrix with shape of (dim, 1)
       b(int): with value of 0
    '''
    w = np.zeros((dim,1))
    b = 0    
    return w, b

def plot_cost_function(costs, learning_rate):
   '''
      Plot learning curve (with costs)
   arguments:
      costs(list): costs per iteration
      learning_rate(int): learning rate of model
   returns:
   '''
   costs = np.squeeze(costs)
   plt.plot(costs)
   plt.ylabel('cost')
   plt.xlabel('iterations (per hundreds)')
   plt.title("Learning rate =" + str(learning_rate))
   plt.show()