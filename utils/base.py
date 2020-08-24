import numpy as np
import matplotlib.pyplot as plt

def initialize_with_zeros(dim):
    '''
     initialize weigths and bias unit with zero
    '''
    w = np.zeros((dim,1))
    b = 0    
    return w, b

def plot_cost_function(costs, learning_rate):
    # Plot learning curve (with costs)
    costs = np.squeeze(costs)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()