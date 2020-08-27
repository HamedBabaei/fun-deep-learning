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
   '''
      Plot image and its labels
   arguments:
      image(image): an image
      label(string): an image label
   returns:
   '''
   plt.imshow(image)
   print ("y = " + str(label))

def flatten_X(X):
   '''
      Convert image into flattend shape
      Image (b=count, c=width, d=height, a=channels(rgb = 3)) where 
      Flatted image Xi is (b*c*d, a)
   arguments:
      X(images): 200 images with shape of (64,64,3) have X with shape of (200, 64, 64, 3)
   returns:
      x_flatten: a flatten image with shape of (200*64*64, 3)
   '''
   x_flatten = X.reshape(X.shape[0], -1).T
   return x_flatten

def standardize_dataset(X):
   '''
     This will first flatten X and next will normalized it by devide into 255
   arguments:
      X(images): with shape of Image (b=count, c=width, d=height, a=channels(rgb = 3))
   returns:
      standard: flatted and normalized images
   '''
   x_flatten = flatten_X(X)
   standard = normalization.normalize_image(x_flatten)
   return standard
   