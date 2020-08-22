'''
Implementation of Logistic Regression with Neural Network Mindset
'''
import __init__
import numpy as np
from utils.activation import sigmoid
from utils.base import initialize_with_zeros

class LogisticRegression:
    def __init__(self):
        pass

    def __propagate(self, w, b, X, Y):
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
    

    def model(self, X_train, Y_train, X_test, Y_test, 
             num_iterations = 2000, learning_rate = 0.5, verbose = False):

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
        return self.d

        def predict(self, X):
            return self.__predict(self.d['w'], self.d['b'], X)


