
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