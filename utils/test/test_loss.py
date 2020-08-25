import __init__
import unittest
import loss
import numpy as np

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])

class TestLossMethods(unittest.TestCase):

    def test_L1(self):
        self.assertEqual(loss.L1(yhat, y) , 1.1)
        
    def test_L2(self):
        self.assertEqual(loss.L2(yhat, y) , 0.43)


if __name__ == '__main__':
    unittest.main()