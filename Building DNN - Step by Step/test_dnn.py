import unittest
import numpy as np
import dnn

class TestDNNMethods(unittest.TestCase):

    def test_initialize_parameters(self):
        n_x, n_h, n_y = 3, 2, 1
        params = dnn.initialize_parameters(n_x, n_h, n_y)
        w1, b1, w2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
        self.assertEqual(w1.shape, (n_h, n_x))
        self.assertEqual(b1.shape, (n_h, 1))
        self.assertEqual(w2.shape, (n_y, n_h))
        self.assertEqual(b2.shape, (n_y, 1))
    
    def test_initialize_parameters_deep(self):
        n_x, n_h, n_y = 3, 2, 1
        params = dnn.initialize_parameters_deep([n_x, n_h, n_y])
        w1, b1, w2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
        self.assertEqual(w1.shape, (n_h, n_x))
        self.assertEqual(b1.shape, (n_h, 1))
        self.assertEqual(w2.shape, (n_y, n_h))
        self.assertEqual(b2.shape, (n_y, 1))
    
    def test_linear_forward(self):
        A = np.array([[ 1.62434536, -0.61175641],[-0.52817175, -1.07296862],[ 0.86540763, -2.3015387 ]])
        W = np.array([[ 1.74481176, -0.7612069 ,  0.3190391 ]])
        b = np.array([[-0.24937038]])
        Z, linear_cache = dnn.linear_forward(A, W, b)
        self.assertEqual(Z.any(), np.array([[ 3.26295337, -1.23429987]]))
    
    def test_linear_activation_forward(self):
        A_prev = np.array([[-0.41675785, -0.05626683],[-2.1361961 ,  1.64027081],[-1.79343559, -0.84174737]])
        W = np.array([[ 0.50288142, -1.24528809, -1.05795222]])
        b = np.array([[-0.90900761]])
        A, linear_activation_cache = dnn.linear_activation_forward(A_prev, W, b, activation='sigmoid')
        self.assertEqual(A.any(), np.array([[0.96890023, 0.11013289]]))
        A, linear_activation_cache = dnn.linear_activation_forward(A_prev, W, b, activation='relu')
        self.assertEqual(A.any(), np.array([[3.43896131, 0.]]))


if __name__ == '__main__':
    unittest.main()
