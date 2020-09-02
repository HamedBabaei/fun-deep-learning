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
    
    def test_forward_propagation(self):
        np.random.seed(1)
        X = np.random.randn(2, 3)
        parameters = {'W1': np.array([[-0.00416758, -0.00056267],[-0.02136196,  0.01640271],
                                      [-0.01793436, -0.00841747],[ 0.00502881, -0.01245288]]),
                      'W2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
                      'b1': np.random.randn(4,1), 
                      'b2': np.array([[ -1.3]])}
        A2, cache = nn.forward_propagation(X, parameters)
        self.assertEqual(A2.shape , (1, X.shape[1]))
        self.assertEqual(np.mean(cache['Z1']), 0.26281864019752443)
        self.assertEqual(np.mean(cache['A1']), 0.09199904522700113)
        self.assertEqual(np.mean(cache['Z2']), -1.3076660128732143)
        self.assertEqual(np.mean(cache['A2']), 0.21287768171914198)
    
    def test_compute_cost(self):
        np.random.seed(1)
        Y = (np.random.randn(1, 3) > 0)
        A2 = (np.array([[ 0.5002307,  0.49985831,  0.50023963]]))
        cost = nn.compute_cost(A2, Y)
        self.assertEqual(cost, 0.6930587610394646)
  


if __name__ == '__main__':
    unittest.main()

