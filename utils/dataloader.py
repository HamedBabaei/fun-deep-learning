import numpy as np
import os
import h5py
        
def load_dataset(path_to_dataset, dataset_name):
   '''
       Data Loader main method for loading datasets
    arguments:
       path_to_dataset(string) 
       dataset_name(string)
    return:
       dataset
   '''
   if dataset_name == 'catvnotcat':
        return __load_catvsnotcat(path_to_dataset)
   if dataset_name == 'planer':
      return __load_planer_dataset()


def __load_catvsnotcat(path_to_dataset):
   '''
      loading catvsnotcat dataset
   arguments:
      path_to_dataset(string) 
   return:
      x_train, y_train, x_test, y_test, classes
   '''
   train_dataset = h5py.File(os.path.join(path_to_dataset, 'train_catvnoncat.h5'), "r")
   train_set_x_orig = np.array(train_dataset["train_set_x"][:]) 
   train_set_y_orig = np.array(train_dataset["train_set_y"][:]) 

   test_dataset = h5py.File(os.path.join(path_to_dataset, 'test_catvnoncat.h5'), "r")
   test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
   test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

   classes = np.array(test_dataset["list_classes"][:]) # the list of classes
   
   train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
   test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
   
   return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def __load_planer_dataset():
   '''
      Loading Planer Dataset
   Arguments:
   Returns:
       X : in a shape of (400,2)
       Y : in a shape of (400,1)
   '''
   np.random.seed(1)
   m = 400 # number of examples
   N = int(m/2) # number of points per class
   D = 2 # dimensionality
   X = np.zeros((m,D)) # data matrix where each row is a single example
   Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
   a = 4 # maximum ray of the flower

   for j in range(2):
      ix = range(N*j,N*(j+1))
      t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
      r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
      X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
      Y[ix] = j
      
   X = X.T
   Y = Y.T
   return X, Y
