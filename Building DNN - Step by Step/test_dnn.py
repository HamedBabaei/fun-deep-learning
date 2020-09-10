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
        self.assertEqual(Z.any(), np.array([[ 3.26295337, -1.23429987]]).any())
    
    def test_linear_activation_forward(self):
        A_prev = np.array([[-0.41675785, -0.05626683],[-2.1361961 ,  1.64027081],[-1.79343559, -0.84174737]])
        W = np.array([[ 0.50288142, -1.24528809, -1.05795222]])
        b = np.array([[-0.90900761]])
        A, linear_activation_cache = dnn.linear_activation_forward(A_prev, W, b, activation='sigmoid')
        self.assertEqual(A.any(), np.array([[0.96890023, 0.11013289]]).any())
        A, linear_activation_cache = dnn.linear_activation_forward(A_prev, W, b, activation='relu')
        self.assertEqual(A.any(), np.array([[3.43896131, 0.]]).any())
    
    def test_L_model_forward(self):
        X = np.array([[-0.31178367,  0.72900392,  0.21782079, -0.8990918 ],
                    [-2.48678065,  0.91325152,  1.12706373, -1.51409323],
                    [ 1.63929108, -0.4298936 ,  2.63128056,  0.60182225],
                    [-0.33588161,  1.23773784,  0.11112817,  0.12915125],
                    [ 0.07612761, -0.15512816,  0.63422534,  0.810655  ]])
        parameters = {
            'W1':np.array([ [ 0.35480861,  1.81259031, -1.3564758 , -0.46363197,  0.82465384],
                            [-1.17643148,  1.56448966,  0.71270509, -0.1810066 ,  0.53419953],
                            [-0.58661296, -1.48185327,  0.85724762,  0.94309899,  0.11444143],
                            [-0.02195668, -2.12714455, -0.83440747, -0.46550831,  0.23371059]]),
            'W2':np.array([ [-0.12673638, -1.36861282,  1.21848065, -0.85750144],
                            [-0.56147088, -1.0335199 ,  0.35877096,  1.07368134],
                            [-0.37550472,  0.39636757, -0.47144628,  2.33660781]]),
            'W3':np.array([[ 0.9398248 ,  0.42628539, -0.75815703]]),
            'b1':np.array([[ 1.38503523], [-0.51962709], [-0.78015214],[ 0.95560959]]),
            'b2':np.array([[ 1.50278553], [-0.59545972], [ 0.52834106]]),
            'b3':np.array([[-0.16236698]])
            }
        AL, caches = dnn.L_model_forward(X, parameters)
        self.assertEqual(len(caches), 3)
        self.assertEqual(AL.any(), np.array([[0.03921668, 0.70498921, 0.19734387, 0.04728177]]).any())
    
    def test_compute_cost(self):
        Y, AL = np.array([[1, 1, 0]]), np.array([[0.8, 0.9, 0.4]])
        cost = dnn.compute_cost(AL, Y)
        self.assertEqual(cost, 0.2797765635793422)
        
if __name__ == '__main__':
    unittest.main()
