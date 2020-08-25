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

