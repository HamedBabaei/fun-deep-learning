import matplotlib.pyplot as plt
import utils.normalization as normalization

def image2vector(image):
    '''
       reshaping image vector
    arguments:
       image: numpy array of shape(length, height, depth)
    returns:
       vector: a vector of shape (length*height*depth, 1)
    '''
    vector = image.reshape((image.shape[0]*image.shape[1]*image.shape[2], 1))
    return vector

def display_image(image, label):
   plt.imshow(image)
   print ("y = " + str(label))

def flatten_X(X):
   x_flatten = X.reshape(X.shape[0], -1).T
   return x_flatten

def standardize_dataset(X):
   '''
     This will first flatten X and next will normalized
   '''
   x_flatten = flatten_X(X)
   standard = normalization.normalize_image(x_flatten)
   return standard
   