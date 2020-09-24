
import matplotlib.pyplot as plt
import scipy.io

def load_2D_dataset(path, plot=False):
    '''
        Loading 2d Dataset 
    Arguments:
        path(str): path to 2d dataset
        plot(bool): to whatever plot dataset or not
    Returns:
        train_X, train_Y, test_X, test_Y
    '''
    data = scipy.io.loadmat(path)
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T
    if plot:
        plt.scatter(train_X[0, :], train_X[1, :], c=train_Y[0], s=40, cmap=plt.cm.Spectral);
    return train_X, train_Y, test_X, test_Y