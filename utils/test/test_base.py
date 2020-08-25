import __init__
import unittest
import numpy as np
import base

class TestBaseMethods(unittest.TestCase):

    def test_initialize_with_zeros(self):
        w, b = base.initialize_with_zeros(2)
        self.assertEqual(w.any(), np.array([[0],[0]]).any())
        self.assertEqual(b, 0)


if __name__ == '__main__':
    unittest.main()