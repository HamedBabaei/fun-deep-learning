'''
Implementation of Logistic Regression with Neural Network Mindset
'''
import numpy as np
import __init__
from utils.activation import sigmoid
from utils.image import flatten_X
from utils.base import initialize_with_zeros
from PIL import Image
from scipy import ndimage
import scipy.misc
import imageio

class LogisticRegression:
    
    def propagate(self, w, b, X, Y):
        '''
            Forward and backward propagation implementation
        Arguments:
            w(numpy): for hodling weight with shape of (num_px * num_px * 3, 1)
            b(int)  : bias a scalar
            X       : data of size (num_px * num_px * 3, number of examples)
            Y       : true labels
        Returns:
            grads: dictionary with following keys:
                dw  : gradient of the loss with respect to w, thus same shape as w
                db  : gradient of the loss with respect to b, thus same shape as b
            cost: negative log-likelihood cost for logistic regression
        '''
        #Forward propagation
        m = X.shape[1]
        A = sigmoid(np.dot(w.T,X) + b)
        cost = (-1/m)*np.sum( Y*np.log(A) + (1-Y)*np.log(1-A) )    
        #backward propagation
        dw = (1/m)*np.dot(X, (A - Y).T)
        db = (1/m)*np.sum(A-Y)
        cost = np.squeeze(cost)        
        grads = {"dw": dw,"db": db}
        return grads, cost

    def optimize(self, w, b, X, Y, num_iterations, learning_rate, verbose = False):
        '''
           train model to optmize its weights
        Arguments:
           w(numpy): for hodling weight with shape of (num_px * num_px * 3, 1)
           b(int)  : bias a scalar
           X       : data of size (num_px * num_px * 3, number of examples)
           Y       : true labels
           num_iterations
           learning_rate
           verbose(bool): to dipslay costs after each iteration
        Returns:
           params: {"w": w, "b": b}
           grads: {"dw": dw,"db": db}
           costs: list of costs per 100 iterations
        '''
        costs = []
        for i in range(num_iterations):
            grads, cost = self.propagate(w, b, X, Y)
            dw = grads["dw"]
            db = grads["db"]
            w = w - learning_rate * dw
            b = b - learning_rate * db
            if i % 100 == 0:
                costs.append(cost)
            if verbose and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
        
        params = {"w": w, "b": b}
        grads = {"dw": dw,"db": db}
        return params, grads, costs
    
    def predict_(self, w, b, X):
        '''
          Make a prediction using pre-trained weights
        Arguments:
          w(numpy): for hodling weight with shape of (num_px * num_px * 3, 1)
          b(int)  : bias a scalar
          X       : data of size (num_px * num_px * 3, number of examples)
        Returns:
          Y_predictions(int): a prediction of the model
        '''
        m = X.shape[1]
        Y_prediction = np.zeros((1,m))
        w = w.reshape(X.shape[0], 1)
        A = sigmoid(np.dot(w.T,X) + b) 
        #unvectorized way
        #for i in range(A.shape[1]):
        #    if A[0,i] <= 0.5:
        #        Y_prediction[0,i] = 0
        #    else:
        #        Y_prediction[0,i] = 1

        # and vectorized way
        Y_prediction[0,:] = 1
        Y_prediction[0, A[0,:] <= 0.5] = 0
        return Y_prediction
    

    def fit(self, X_train, Y_train, X_test, Y_test, 
             num_iterations = 2000, learning_rate = 0.5, 
             verbose = False, output_pretrained_parameters = False):
        '''
           Creating LogisticRegression model with Neural Network Mindset
        Arguments:
           X_train                      : a numpy array of shape (num_px * num_px * 3, m_train)
           Y_train                      : a numpy vector of shape (1, m_train)
           X_test                       : a numpy array of shape (num_px * num_px * 3, m_test)
           Y_test                       : a numpy vector of shape (1, m_test)
           num_iterations               : train iterations
           learning_rate                : a learning rate of model,
           verbose                      : display training status
           output_pretrained_parameters : return model pretrained parameters
        returns:
           d(dict): a dictionary containing information about the model
        '''
        w, b = initialize_with_zeros(X_train.shape[0])

        parameters, grads, costs = self.optimize(w, b, X_train, Y_train, 
                                                   num_iterations= num_iterations, 
                                                   learning_rate = learning_rate, 
                                                   verbose= verbose)

        w = parameters["w"]
        b = parameters["b"]
        Y_prediction_test = self.predict_(w, b, X_test)
        Y_prediction_train = self.predict_(w, b, X_train)

        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

        self.d = {"costs": costs, 
                  "Y_prediction_test": Y_prediction_test, 
                  "Y_prediction_train" : Y_prediction_train, 
                  "w" : w, "dw":grads['dw'],  
                  "b" : b, "db":grads['db'],
                  "learning_rate" : learning_rate, 
                  "num_iterations": num_iterations}

        if output_pretrained_parameters:
            return self.d

    def predict(self, X):
        '''
            A general method for calling from user side to make prediction
        Arguments:
            X: input features
        Returns:
            prediction(int): model predictions
        '''
        return self.predict_(self.d['w'], self.d['b'], X)
    
    def predict_by_image(self, path_to_image, dim = (64,64)):
        '''
           prediction by preprocessing for prediction
        Arguments:
           path_to_image
           dim(touple): for reszing image
        returns:
           predict_
        '''
        image = np.array(imageio.imread(path_to_image))  # read a standard image
        image = image/255.
        image.resize((dim[1]*dim[1]*3, 1))
        return self.predict(image)[0]
