import sys
sys.path.append("..")
import unittest
import numpy as np
import activation

sigmoid_inp_x = np.array([1, 2, 3])
sigmoid_out_x = np.array([0.73105858, 0.88079708, 0.95257413])
sigmoid_out_ds = np.array([0.19661193, 0.10499359, 0.04517666])        
softmax_inp_x = np.array([[9, 2, 5, 0, 0],[7, 5, 0, 0 ,0]])
softmax_out_x = np.array([[9.80897665e-01, 8.94462891e-04, 1.79657674e-02, 1.21052389e-04, 1.21052389e-04],
                    [8.78679856e-01, 1.18916387e-01, 8.01252314e-04, 8.01252314e-04, 8.01252314e-04]])

class TestActivationMethods(unittest.TestCase):

    def test_sigmoid(self):
        self.assertEqual(activation.sigmoid(3), 0.9525741268224334)
        self.assertEqual(activation.sigmoid(sigmoid_inp_x).shape, sigmoid_out_x.shape)
        self.assertEqual(activation.sigmoid(sigmoid_inp_x).any(), sigmoid_out_x.any())

    def test_sigmoid_derivative(self):
        self.assertEqual(activation.sigmoid_derivative(sigmoid_inp_x).any(), sigmoid_out_ds.any())
        
    def test_softmax(self):
        self.assertEqual(activation.softmax(softmax_inp_x).any(), softmax_out_x.any())

if __name__ == '__main__':
    unittest.main()