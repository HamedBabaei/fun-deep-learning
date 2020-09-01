import unittest
import numpy as np
import nn

class TestNNMethods(unittest.TestCase):

    def test_layer_size(self):
        np.random.seed(1)
        X = np.random.randn(5,3)
        Y = np.random.randn(2,3)
        n_x, n_h, n_y = nn.layer_size(X, Y)
        self.assertEqual(n_x, 5)
        self.assertEqual(n_h, 4)
        self.assertEqual(n_y, 2)
    
    def test_initialize_parameters(self):
        n_x, n_h, n_y = 2, 4, 1
        params = nn.initialize_parameters(n_x, n_h, n_y)
        w1, b1, w2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
        self.assertEqual(w1.shape, (n_h, n_x))
        self.assertEqual(b1.shape, (n_h, 1))
        self.assertEqual(w2.shape, (n_y, n_h))
        self.assertEqual(b2.shape, (n_y, 1))




if __name__ == '__main__':
    unittest.main()

