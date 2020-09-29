import numpy as np
import activations
import matplotlib.pyplot as plt

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
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L): 
        #parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
    return parameters

def initialize_parameters_zeros(layer_dims):
    '''
        Initialize Deep Neural Network Parameters with zero
    Arguments:
        layer_dims: list of units for each layers
    Returns:
        parameters: a dictionary containing parameters "W1", "b1",......"WL","bL"
    '''
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L): 
        parameters['W' + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
    return parameters

def initialize_parameters_random(layer_dims):
    '''
        Initialize Deep Neural Network Parameters with random values
    Arguments:
        layer_dims: list of units for each layers
    Returns:
        parameters: a dictionary containing parameters "W1", "b1",......"WL","bL"
    '''
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L): 
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.1
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
    return parameters

def initialize_parameters_he(layer_dims):
    '''
        Initialize Deep Neural Network Parameters with he method
    Arguments:
        layer_dims: list of units for each layers
    Returns:
        parameters: a dictionary containing parameters "W1", "b1",......"WL","bL"
    '''
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L): 
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*np.sqrt(2./layer_dims[l-1])
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

def dropout(A , keep_prob):
    D = np.random.rand(A.shape[0], A.shape[1])
    D = (D < keep_prob).astype(int)
    A = A*D
    A = A/keep_prob
    return A, D

def linear_activation_forward(A_prev, W, b, activation, keep_prob = 1):
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
    A , dropout_cache = dropout(A, keep_prob)
    cache = (linear_cache, activation_cache, dropout_cache)
    #print("forward cache: A:{}, dropout:{}".format( A.shape, dropout_cache.shape))
    return A, cache

def L_model_forward(X, parameters, keep_prob=1):
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
                                             activation="relu", 
                                             keep_prob=keep_prob)
        caches.append(cache)
        #print("L forward, layer ",str(l),'::',A.shape)
    AL, cache = linear_activation_forward(A,
                                          parameters["W"+str(l+1)], 
                                          parameters["b"+str(l+1)], 
                                          activation="sigmoid")
    caches.append(cache)
    #print("L forward, layer ",str(l),'::',AL.shape)
    return AL, caches

def regularization(parameters, m, lambd):
    regularization_cost = np.sum([np.sum(np.square(parameters['W'+str(l+1)])) for l in range(len(parameters)//2)])
    L2_regularization_cost = (1/m) * (lambd/2) * regularization_cost
    return L2_regularization_cost

def compute_cost(AL, Y, parameters=None, lambd=0):
    '''
        Implement the cost function
    Argument:
        AL: probability vector corresponding to your label predictions in shape of (1, number of examples)
        Y: truth label
        parameters: python dictionary containing parameters {W1: -, b1: -, ....}
        lambd: a regularization term when its none 0 regularizaton will effect
    Returns:
        cost: cross entropy cost or regularized loss function
    '''
    m = Y.shape[1]
    #cross_entropy_cost = (-1/m)*np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL))
    logprobs = np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1 - AL), 1 - Y)
    cross_entropy_cost = 1./m*np.nansum(logprobs)
    #cross_entropy_cost = np.squeeze(cross_entropy_cost)
    if lambd != 0:
        L2_regularization_cost = regularization(parameters, m, lambd)
        cost = cross_entropy_cost + L2_regularization_cost
        #print(regularization_cost, '  ', lambd)
    else:
        cost = cross_entropy_cost
    return cost

def linear_backward(dZ, cache):
    '''
        Implement the linear portion of backward propagation
    Arguments:
        dZ:  gradient of the cost with respect to the linear output
        cache: touple of (A_prev, W, b)
    returns:
        dA_prev, gradient of the cost with respect to A
        dW,  gradient of the cost with respect to w
        db,  gradient of the cost with respect to b
    '''
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1./m)*np.dot(dZ, A_prev.T)
    db = (1./m)*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation, L2_regularization, keep_prob):
    '''
        Implement the backward propagation for the LINEAR->ACTIVATION layer
    Arguments:
        dA: post activation gradient for current layer l
        cache: tuple of values (linear_cache, activation_cache)
        activation: activation to be used in this layer (relu or sigmoid)
    Returns:
        dA_prev: gradient of the cost with respect to activaton of previews layer (l-1)
        dW: gradient of the cost with respect to W
        db: gradient of the cost with respect to b
    '''
    acts = {"sigmoid":activations.sigmoid_backward, "relu":activations.relu_backward}
    linear_cache, activation_cache, dropout_cache = cache
    #dropout
    if keep_prob != 1:
        #print('dA: ',dA.shape, '   D:', dropout_cache.shape)
        dA = dA * dropout_cache
        dA = dA/keep_prob
    dZ = acts[activation](dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    #regularization
    dW += L2_regularization*linear_cache[1] # linear_cache[1] is W
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches, L2_regularization, keep_prob):
    '''
        Implement the backward propogation for the [LINEAR->RELU]*(L-1) -> LINEAR -> SIGMOID group
    Arguments:
        AL: probability vector of the L_model_forward()
        Y: true label vector
        caches: list of cahces containing lists of ((A, W, b), Z)
        lambd: a regularization value
    Returns:
        grads: a dictionary with the gradients {'dA1': -, 'dW1': -, 'db1': -}
    '''
    grads = {}
    L = len(caches)
    #m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    #Initialize backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    # Lth layer (SIGMOID->LINEAR)
    current_cache = caches[L-1]
    grads['dA'+str(L-1)], grads['dW'+str(L)], grads['db'+str(L)] = \
                                            linear_activation_backward(dAL, 
                                                                       current_cache, 
                                                                       activation='sigmoid',
                                                                       L2_regularization=0, 
                                                                       keep_prob=1)
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_tmp, dW_tmp, db_tmp = linear_activation_backward(grads['dA'+str(l+1)], 
                                                                 current_cache, 
                                                                 activation='relu', 
                                                                 L2_regularization=L2_regularization,
                                                                 keep_prob=keep_prob)
        grads['dA'+str(l)] = dA_prev_tmp
        grads['dW'+str(l+1)] = dW_tmp
        grads['db'+str(l+1)] = db_tmp
    return grads

def update_parameters(parameters, grads, learning_rate):
    '''
        Update parameters using gradient descent
    Arguments:
        parameters: python dictionary containing parameters {W1: -, b1: -, ....}
        grads: python dictionary containing gradients output of L_model_backward
    Returns:
        parameters: python dictionary containing updated parameters
    '''
    L = len(parameters)//2
    for l in range(L):
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate*grads['dW'+str(l+1)]
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate*grads['db'+str(l+1)]
    return parameters

def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False, plot_cost=False):
    '''
        Two layer neural network model: LINEAR->RELU->LINEAR->SIGMOID.
    Arguments:
        X: input data in shape of (n_x, number of examples)
        Y: true label in shape of (1, number of examples)
        layers_dims: dimensions of layers(n_x,n_h, n_y)
        num_iterations: number of iterations of the optimization loop
        learning_rate: learning rate of the gradient descent update rule
        print_cost: printing cost per 100 iteration
        plot_cost: ploting costs after finishing training
    Returns:
        parameters: a dictionary containing W1, W2, b1, and b2
    '''
    np.random.seed(1)
    grads = {}
    costs = []
    m = X.shape[1]
    (n_x, n_h, n_y) = layers_dims

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    for i in range(0, num_iterations):

        A1, cache1 = linear_activation_forward(X, W1, b1, activation='relu')
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation='sigmoid')

        cost = compute_cost(A2, Y)

        dA2 = -(np.divide(Y, A2) - np.divide(1-Y, 1-A2))

        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation='sigmoid')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation='relu')

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if (print_cost and i % 100 == 0) or plot_cost:
            costs.append(cost)

    if plot_cost:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    return parameters

def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations=3000, print_cost=False, plot_cost=False, lambd = 0, keep_prob=1, initialization="None"):
    '''
        L layer neural network model: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    Arguments:
        X: input data in shape of (n_x, number of examples)
        Y: true label in shape of (1, number of examples)
        layers_dims: dimensions of layers(n_x,n_h, n_y)
        num_iterations: number of iterations of the optimization loop
        learning_rate: learning rate of the gradient descent update rule
        print_cost: printing cost per 100 iteration
        plot_cost: ploting costs after finishing training
        initialization: initialization parameter method
    Returns:
        parameters: a dictionary containing W1, W2, b1, and b2
    '''
    m = X.shape[1] #for regularization
    np.random.seed(1)
    costs = []
    if  initialization == 'zeros':
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == 'he':
        parameters = initialize_parameters_he(layers_dims)
    elif initialization == 'random':
        parameters = initialize_parameters_random(layers_dims)
    else:
        parameters = initialize_parameters_deep(layers_dims)
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters, keep_prob=keep_prob)
        cost = compute_cost(AL, Y, parameters=parameters, lambd=lambd)
        grads = L_model_backward(AL, Y, caches, L2_regularization = lambd/m, keep_prob=keep_prob)
        
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if (print_cost and i % 100 == 0) or plot_cost:
            costs.append(cost)

    if plot_cost:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    return parameters

def predict(X, parameters):
    '''
        Make Prediction using learned parameters
    Arguments:
        parameters: {"W1":W1, "b1":b1, "W2":W2, "b2":b2}
        X: input data of size (n_x, m)
    Returns:
        predictions: vector of predictions of the model
    ''' 
    probas, caches = L_model_forward(X, parameters)
    predictions = np.zeros((1,X.shape[1]))
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            predictions[0,i] = 1
        else:
            predictions[0,i] = 0
    return predictions

def predict_dec(X, parameters):
    '''
        Make Prediction using learned parameters to plotting decision boundary
    Arguments:
        parameters: {"W1":W1, "b1":b1, "W2":W2, "b2":b2}
        X: input data of size (n_x, m)
    Returns:
        predictions: vector of predictions of the model
    ''' 
    probas, caches = L_model_forward(X, parameters)
    predicts = (probas > 0.5)
    return predicts

class DNN(object):
    def __init__(self, parameters):
        self.parameters = parameters

    def fit(self):
        pass

    def model(self):
        pass

    def predict(self, X):
        '''
            Make Prediction using learned parameters
        Arguments:
            X: input data of size (n_x, m)
        Returns:
            predictions: vector of predictions of the model
        '''
        return predict(X, self.parameters)
    
    def predict_dec(self, X):
        '''
            Make Prediction using learned parameters
        Arguments:
            X: input data of size (n_x, m)
        Returns:
            predictions: vector of predictions of the model
        '''
        return predict_dec(X, self.parameters)

class TwoLayerModel(DNN):
   
    def __init__(self, layers_dims, num_iterations = 10000, learning_rate = 0.0075):
        '''
            Neural Network Class init
        Arguments:
            hidden_layer_size: size of the hidden layer
            num_iterations: number of iterations in gradient descent loop
            learning_rate: learning rate
        Return:
            None
        '''
        self.layers_dims = layers_dims
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        DNN.__init__(self, parameters=None)
   
    def fit(self, X, Y, verbose = False, plot_cost=False):
        '''
            Neural Network fit module
        Arguments:
            X : datasate of shape (number of features, number of examples)
            Y : labels of shape  (1, number of examples)
            verbose: print costs per 100 iteration
            plot_cost: to plot costs after finishing training
        Return:
            None
        '''
        self.parameters = two_layer_model(X, Y, self.layers_dims, 
                                    num_iterations = self.num_iterations,
                                    print_cost=verbose, 
                                    learning_rate = self.learning_rate,
                                    plot_cost=plot_cost)
                                    
    def model(self, X, Y, verbose=False):
        return two_layer_model(X, Y, self.layers_dims, 
                        num_iterations = self.num_iterations,
                        print_cost=verbose, 
                        learning_rate = self.learning_rate)

class LLayerModel(DNN):    
    
    def __init__(self, layers_dims, num_iterations = 10000, learning_rate = 0.0075, lambd = 0, keep_prob = 1, initialization='None'):
        '''
            Neural Network Class init
        Arguments:
            hidden_layer_size: size of the hidden layer
            num_iterations: number of iterations in gradient descent loop
            learning_rate: learning rate
        Return:
            None
        '''
        self.layers_dims = layers_dims
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.lambd = lambd
        self.keep_prob = keep_prob
        self.initialization = initialization
        DNN.__init__(self, parameters=None)
    
    def fit(self, X, Y, verbose = False, plot_cost=False):
        '''
            Neural Network fit module
        Arguments:
            X : datasate of shape (number of features, number of examples)
            Y : labels of shape  (1, number of examples)
            verbose: print costs per 100 iteration
            plot_cost: to plot costs after finishing training
        Return:
            None
        '''
        self.parameters = L_layer_model(X, Y, self.layers_dims, 
                                        num_iterations = self.num_iterations,
                                        print_cost=verbose, 
                                        learning_rate=self.learning_rate,
                                        plot_cost=plot_cost,
                                        lambd=self.lambd,
                                        keep_prob=self.keep_prob,
                                        initialization=self.initialization)

    def model(self, X, Y, verbose=False):
        return L_layer_model(X, Y, self.layers_dims, 
                        num_iterations = self.num_iterations,
                        print_cost=verbose, 
                        learning_rate = self.learning_rate,
                        lambd=self.lambd,
                        keep_prob=self.keep_prob,
                        initialization=self.initialization)

