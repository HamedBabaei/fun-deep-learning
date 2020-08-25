'''
Implementation of Logistic Regression with Neural Network Mindset
'''
import numpy as np
import __init__
from utils.activation import sigmoid
from utils.base import initialize_with_zeros

class LogisticRegression:
    def __propagate(self, w, b, X, Y):
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

    def __optimize(self, w, b, X, Y, num_iterations, learning_rate, verbose = False):
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
           params:
           grads:
           costs:
        '''
        costs = []
        for i in range(num_iterations):
            grads, cost = self.__propagate(w, b, X, Y)
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

    def __predict(self, w, b, X):
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
        for i in range(A.shape[1]):
            if A[0,i] <= 0.5:
                Y_prediction[0,i] = 0
            else:
                Y_prediction[0, i] = 1

        return Y_prediction
    

    def fit(self, X_train, Y_train, X_test, Y_test, 
             num_iterations = 2000, learning_rate = 0.5, 
             verbose = False, output_pretrained_parameters = False):

        w, b = initialize_with_zeros(X_train.shape[0])

        parameters, grads, costs = self.__optimize(w, b, X_train, Y_train, 
                                                   num_iterations= num_iterations, 
                                                   learning_rate = learning_rate, 
                                                   verbose= verbose)

        w = parameters["w"]
        b = parameters["b"]
        Y_prediction_test = self.__predict(w, b, X_test)
        Y_prediction_train = self.__predict(w, b, X_train)

        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

        self.d = {"costs": costs, "Y_prediction_test": Y_prediction_test, 
            "Y_prediction_train" : Y_prediction_train, "w" : w,  "b" : b,
            "learning_rate" : learning_rate, "num_iterations": num_iterations}
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
        return self.__predict(self.d['w'], self.d['b'], X)


