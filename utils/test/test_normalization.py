import sys
sys.path.append("..")
import unittest
import normalization
import numpy as np

class TestNormalizationMethods(unittest.TestCase):

    def test_normalize_rows(self):
        x = np.array([[0, 3, 4],[1, 6, 4]])
        out = np.array([[0.0, 0.6, 0.8], [0.13736056, 0.82416338, 0.54944226]])
        self.assertEqual(normalization.normalize_rows(x).shape, out.shape)
        self.assertEqual(normalization.normalize_rows(x).any(), out.any())

if __name__ == '__main__':
    unittest.main()