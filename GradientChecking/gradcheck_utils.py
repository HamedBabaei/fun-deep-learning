import numpy as np
import dnn

def gradient_check(parameters, gradients, X, Y, epsilon = 1e-7):
    '''
        Gradient Checking method
    Arguments:
        parameters: a dictionary containing W1, b1, W2, b2, ....
        gradients: a dictionary contaning dW1, db1, dW2, db2, ...
        X: a training data
        Y: truth labels
        epsilon: for gradient checking
    Returns:
        differences
    '''
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    for i in range(num_parameters):
    
        thetaplus = np.copy(parameters_values)            
        thetaplus[i][0] = thetaplus[i][0] + epsilon       
        AL, _ = dnn.L_model_forward(X, vector_to_dictionary(thetaplus))
        J_plus[i] = dnn.compute_cost(AL, Y) 
        
        thetaminus = np.copy(parameters_values)           
        thetaminus[i][0] = thetaminus[i][0] - epsilon     
        AL, _ = dnn.L_model_forward(X, vector_to_dictionary(thetaminus))
        J_minus[i] = dnn.compute_cost(AL, Y) 

        gradapprox[i] = (J_plus[i] - J_minus[i])/(2*epsilon)
    
    
    numerator = np.linalg.norm(grad - gradapprox)                   
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox) 
    difference = numerator/denominator   
    
    if difference > 2e-7:
        print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    
    return difference

def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    count = 0
    for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        
        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1,1))
        keys = keys + [key]*new_vector.shape[0]
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, keys

def vector_to_dictionary(theta):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}
    parameters["W1"] = theta[:20].reshape((5,4))
    parameters["b1"] = theta[20:25].reshape((5,1))
    parameters["W2"] = theta[25:40].reshape((3,5))
    parameters["b2"] = theta[40:43].reshape((3,1))
    parameters["W3"] = theta[43:46].reshape((1,3))
    parameters["b3"] = theta[46:47].reshape((1,1))

    return parameters

def gradients_to_vector(gradients):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """
    
    count = 0
    for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3"]:
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1,1))
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta