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

    def test_backward_propogation(self):
        np.random.seed(1)
        X_assess = np.random.randn(2, 3)
        Y_assess = (np.random.randn(1, 3) > 0)
        
        parameters = {'W1': np.array([[-0.00416758, -0.00056267],[-0.02136196,  0.01640271],
                                      [-0.01793436, -0.00841747],[ 0.00502881, -0.01245288]]),
                      'W2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
                      'b1': np.array([[ 0.], [ 0.], [ 0.], [ 0.]]),
                      'b2': np.array([[ 0.]])}

        cache = {'A1': np.array([[-0.00616578,  0.0020626 ,  0.00349619],
                                 [-0.05225116,  0.02725659, -0.02646251],
                                 [-0.02009721,  0.0036869 ,  0.02883756],
                                 [ 0.02152675, -0.01385234,  0.02599885]]),
                 'A2': np.array([[ 0.5002307 ,  0.49985831,  0.50023963]]),
                 'Z1': np.array([[-0.00616586,  0.0020626 ,  0.0034962 ],
                                 [-0.05229879,  0.02726335, -0.02646869],
                                 [-0.02009991,  0.00368692,  0.02884556],
                                 [ 0.02153007, -0.01385322,  0.02600471]]),
                 'Z2': np.array([[ 0.00092281, -0.00056678,  0.00095853]])}
        #grads
        dW1 = np.array([[ 0.00301023, -0.00747267], [ 0.00257968, -0.00641288],
                        [-0.00156892,  0.003893  ], [-0.00652037,  0.01618243]])
        db1 = np.array([[ 0.00176201], [ 0.00150995], [-0.00091736], [-0.00381422]])
        dW2 = np.array([[ 0.00078841,  0.01765429, -0.00084166, -0.01022527]])
        db2 = np.array([[-0.16655712]])

        grads = nn.backward_propagation(parameters, cache, X_assess, Y_assess)
        self.assertEqual(dW1.any(), grads['dW1'].any())
        self.assertEqual(dW2.any(), grads['dW2'].any())
        self.assertEqual(db1.any(), grads['db1'].any())
        self.assertEqual(db2.any(), grads['db2'].any())
    
    def test_update_parameters(self):
        parameters = {'W1': np.array([[-0.00615039,  0.0169021 ],[-0.02311792,  0.03137121],
                                       [-0.0169217 , -0.01752545],[ 0.00935436, -0.05018221]]),
                      'W2': np.array([[-0.0104319 , -0.04019007,  0.01607211,  0.04440255]]),
                      'b1': np.array([[ -8.97523455e-07],[  8.15562092e-06],[  6.04810633e-07],[ -2.54560700e-06]]),
                      'b2': np.array([[  9.14954378e-05]])}
        grads = {'dW1': np.array([[ 0.00023322, -0.00205423],[ 0.00082222, -0.00700776],
                                  [-0.00031831,  0.0028636 ],[-0.00092857,  0.00809933]]),
                 'dW2': np.array([[ -1.75740039e-05,   3.70231337e-03,  -1.25683095e-03,-2.55715317e-03]]),
                 'db1': np.array([[  1.05570087e-07],[ -3.81814487e-06],[ -1.90155145e-07],[  5.46467802e-07]]),
                 'db2': np.array([[ -1.08923140e-05]])}
        rs_params ={"W1": np.array([[-0.00643025,  0.01936718], [-0.02410458,  0.03978052],
                                     [-0.01653973, -0.02096177], [ 0.01046864, -0.05990141]]),
                    "b1": np.array([[-1.02420756e-06],[1.27373948e-05],[8.32996807e-07],[-3.20136836e-06]]),
                    "W2": np.array([[-0.01041081, -0.04463285, 0.01758031, 0.04747113]]),
                    "b2": np.array([[0.00010457]])}
        params = nn.update_parameters(parameters, grads)
        self.assertEqual(rs_params['W1'].any(), params['W1'].any())
        self.assertEqual(rs_params['W2'].any(), params['W2'].any())
        self.assertEqual(rs_params['b1'].any(), params['b1'].any())
        self.assertEqual(rs_params['b2'].any(), params['b2'].any())

    def test_nn_model(self):
        np.random.seed(1)
        X = np.random.randn(2, 3)
        Y = (np.random.randn(1, 3) > 0)
        params = nn.nn_model(X, Y, 4)
        rs = {"W1": np.array([[-0.65848169, 1.21866811], [-0.76204273,  1.39377573],
                              [ 0.5792005, -1.10397703], [ 0.76773391, -1.41477129]]),
              "b1": np.array([[ 0.287592], [ 0.3511264 ], [-0.2431246 ], [-0.35772805]]),
              "W2": np.array([[-2.45566237, -3.27042274,  2.00784958,  3.36773273]]),
              "b2": np.array([[0.20459656]])}
        self.assertEqual(params['W1'].any(), rs['W1'].any())
        self.assertEqual(params['b1'].any(), rs['b1'].any())
        self.assertEqual(params['W2'].any(), rs['W2'].any())
        self.assertEqual(params['b2'].any(), rs['b2'].any())
    
    def test_predicts(self):
        np.random.seed(1)
        X_assess = np.random.randn(2, 3)
        parameters = {'W1': np.array([[-0.00615039,  0.0169021 ],[-0.02311792,  0.03137121],
                                    [-0.0169217 , -0.01752545],[ 0.00935436, -0.05018221]]),
                    'W2': np.array([[-0.0104319 , -0.04019007,  0.01607211,  0.04440255]]),
                    'b1': np.array([[ -8.97523455e-07], [  8.15562092e-06], [  6.04810633e-07],[ -2.54560700e-06]]),
                    'b2': np.array([[  9.14954378e-05]])}
        preds = nn.predict(parameters, X_assess)
        self.assertEqual(np.mean(preds), 0.6666666666666666)

if __name__ == '__main__':
    unittest.main()

