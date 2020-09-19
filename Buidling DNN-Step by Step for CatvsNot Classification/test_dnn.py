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

    def test_linear_backward(self):
        dZ = np.array([[ 1.62434536, -0.61175641, -0.52817175, -1.07296862],
                        [ 0.86540763, -2.3015387 ,  1.74481176, -0.7612069 ],
                        [ 0.3190391 , -0.24937038,  1.46210794, -2.06014071]])
        A = np.array([[-0.3224172 , -0.38405435,  1.13376944, -1.09989127],
                        [-0.17242821, -0.87785842,  0.04221375,  0.58281521],
                        [-1.10061918,  1.14472371,  0.90159072,  0.50249434],
                        [ 0.90085595, -0.68372786, -0.12289023, -0.93576943],
                        [-0.26788808,  0.53035547, -0.69166075, -0.39675353]])
        W = np.array([[-0.6871727 , -0.84520564, -0.67124613, -0.0126646 , -1.11731035],
                        [ 0.2344157 ,  1.65980218,  0.74204416, -0.19183555, -0.88762896],
                        [-0.74715829,  1.6924546 ,  0.05080775, -0.63699565,  0.19091548]])
        b = np.array([[2.10025514], [0.12015895], [0.61720311]])
        
        A_prev_ans = np.array([[-1.15171336,  0.06718465, -0.3204696 ,  2.09812712],
                                [ 0.60345879, -3.72508701,  5.81700741, -3.84326836],
                                [-0.4319552 , -1.30987417,  1.72354705,  0.05070578],
                                [-0.38981415,  0.60811244, -1.25938424,  1.47191593],
                                [-2.52214926,  2.67882552, -0.67947465,  1.48119548]])
        dW_ans = np.array([[ 0.07313866, -0.0976715 , -0.87585828,  0.73763362,  0.00785716],
                            [ 0.85508818,  0.37530413, -0.59912655,  0.71278189, -0.58931808],
                            [ 0.97913304, -0.24376494, -0.08839671,  0.55151192, -0.10290907]])
        db_ans = np. array([[-0.14713786], [-0.11313155],[-0.13209101]])

        linear_cache = (A, W, b)
        A_prev, dW, db = dnn.linear_backward(dZ, linear_cache)
        self.assertEqual(A_prev_ans.any(), A_prev.any())
        self.assertEqual(dW_ans.any(), dW.any())
        self.assertEqual(db_ans.any(), db.any())
    
    def linear_activation_backward(self):
        dAL = np.array([[-0.41675785, -0.05626683]])
        A = np.array([[-2.1361961 ,  1.64027081], [-1.79343559, -0.84174737],[ 0.50288142, -1.24528809]])
        W = np.array([[-1.05795222, -0.90900761,  0.55145404]])
        b = np.array([[2.29220801]])
        Z = np.array([[ 0.04153939, -1.11792545]])
        
        linear_activation_cache = ((A, W, b), Z)
        
        dA_prev_ans_relu = np.array([[ 0.44090989,  0. ],[ 0.37883606,  0.], [-0.2298228 ,  0.]])
        dW_ans_relu = np.array([[ 0.44513824,  0.37371418, -0.10478989]])
        db_ans_relu = np. array([[-0.20837892]])
        dA_prev_relu, dW_relu, db_relu = dnn.linear_activation_backward(dAL, linear_activation_cache, activation='relu')
        
        self.assertEqual(dW_ans_relu.any(), dW_relu.any())
        self.assertEqual(db_ans_relu.any(), db_relu.any())
        self.assertEqual(dA_prev_ans_relu.any(), dA_prev_relu.any())
        
        dA_prev_ans_sigmoid = np.array([[ 0.11017994,  0.01105339], [ 0.09466817,  0.00949723], [-0.05743092, -0.00576154]])
        dW_ans_sigmoid = np.array([[ 0.10266786,  0.09778551, -0.01968084]])
        db_ans_sigmoid = np.array([[-0.05729622]])
        dA_prev_sigmoid, dW_sigmoid, db_sigmoid = dnn.linear_activation_backward(dAL, linear_activation_cache, activation='sigmoid')

        self.assertEqual(dW_ans_sigmoid.any(), dW_sigmoid.any())
        self.assertEqual(db_ans_sigmoid.any(), db_sigmoid.any())
        self.assertEqual(dA_prev_ans_sigmoid.any(), dA_prev_sigmoid.any())


    def test_L_model_backward(self):
        AL = np.array([[1.78862847, 0.43650985]])
        Y = np.array([[1, 0]])

        A1 = np.array([[ 0.09649747, -1.8634927 ], [-0.2773882 , -0.35475898],
                    [-0.08274148, -0.62700068],[-0.04381817, -0.47721803]])
        W1= np.array([[-1.31386475,  0.88462238,  0.88131804,  1.70957306],
                        [ 0.05003364, -0.40467741, -0.54535995, -1.54647732],
                        [ 0.98236743, -1.10106763, -1.18504653, -0.2056499 ]])
        b1 = np.array([[ 1.48614836], [ 0.23671627], [-1.02378514]])
        Z1 = np.array([[-0.7129932 ,  0.62524497], [-0.16051336, -0.76883635],[-0.23003072,  0.74505627]])
        cache1 = ((A1, W1, b1), Z1)

        A2 = np.array([[ 1.97611078, -1.24412333],[-0.62641691, -0.80376609], [-2.41908317, -0.92379202]])
        W2 = np.array([[-1.02387576,  1.12397796, -0.13191423]])
        b2 = np.array([[-1.62328545]])
        Z2 = np.array([[ 0.64667545, -0.35627076]])
        cache2 = ((A2, W2, b2), Z2)

        caches = (cache1, cache2)
        
        ans_grads = {'dA0': np.array([[ 0.,  0.52257901], [ 0., -0.3269206 ], [ 0., -0.32070404], [ 0., -0.74079187]]), 
                     'dA1': np.array([[ 0.12913162, -0.44014127], [-0.14175655,  0.48317296], [ 0.01663708, -0.05670698]]), 
                     'dW1': np.array([[0.41010002, 0.07807203, 0.13798444, 0.10502167],
                                   [0., 0., 0. , 0.],
                                   [0.05283652, 0.01005865, 0.01777766, 0.0135308 ]]), 
                     'dW2': np.array([[-0.39202432, -0.13325855, -0.04601089]]), 
                     'db1': np.array([[-0.22007063], [ 0.        ], [-0.02835349]]), 
                     'db2': np.array([[0.15187861]])}

        grads = dnn.L_model_backward(AL, Y, caches)

        for d, grad in grads.items():
            self.assertEqual(grad.any(), ans_grads[d].any())

    def test_update_parameters(self):
        params = {'W1': np.array([[-0.59562069, -0.09991781, -2.14584584,  1.82662008],
                            [-1.76569676, -0.80627147,  0.51115557, -1.18258802],
                            [-1.0535704 , -0.86128581,  0.68284052,  2.20374577]]),
                    'W2': np.array([[-0.55569196,  0.0354055 ,  1.32964895]]),
                    'b1': np.array([[-0.04659241],[-1.28888275], [ 0.53405496]]),
                    'b2': np.array([[-0.84610769]])}
        grads = {'dW1': np.array([[ 1.78862847,  0.43650985,  0.09649747, -1.8634927 ],
                                [-0.2773882 , -0.35475898, -0.08274148, -0.62700068],
                                [-0.04381817, -0.47721803, -1.31386475,  0.88462238]]),
                'dW2': np.array([[-0.40467741, -0.54535995, -1.54647732]]),
                'db1': np.array([[0.88131804], [1.70957306], [0.05003364]]),
                'db2': np.array([[0.98236743]])}

        ans_updated_params = {'W1': np.array([[-0.59562069, -0.09991781, -2.14584584,  1.82662008],
                            [-1.76569676, -0.80627147,  0.51115557, -1.18258802],
                            [-1.0535704 , -0.86128581,  0.68284052,  2.20374577]]),
                'W2': np.array([[-0.55569196,  0.0354055 ,  1.32964895]]),
                'b1': np.array([[-0.04659241], [-1.28888275], [ 0.53405496]]),
                'b2': np.array([[-0.84610769]])}

        updated_params = dnn.update_parameters(params, grads, 0.1)

        for p, update in updated_params.items():
            self.assertEqual(update.any(), ans_updated_params[p].any())


if __name__ == '__main__':
    unittest.main()
