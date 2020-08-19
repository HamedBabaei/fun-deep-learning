import sys
sys.path.append("..")
import unittest
import numpy as np
import activation

class TestActivationMethods(unittest.TestCase):

    def test_sigmoid(self):
        #x = np.array([1, 2, 3])
        #self.assertEqual(sigmoid(x), np.array([0.73105858, 0.88079708, 0.95257413]))
        self.assertEqual(activation.sigmoid(3), 0.9525741268224334)

if __name__ == '__main__':
    unittest.main()