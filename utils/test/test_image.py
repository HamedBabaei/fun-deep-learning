import sys
sys.path.append("..")
import unittest
import image
import numpy as np

class TestImageMethods(unittest.TestCase):

    def test_image2vector(self):
        x_img = np.array([[ [ 0.67826139,  0.29380381],
                            [ 0.90714982,  0.52835647],
                            [ 0.4215251 ,  0.45017551]],

                          [ [ 0.92814219,  0.96677647],
                            [ 0.85304703,  0.52351845],
                            [ 0.19981397,  0.27417313]],

                          [ [ 0.60659855,  0.00533165],
                            [ 0.10820313,  0.49978937],
                            [ 0.34144279,  0.94630077]] ])
        y_img = np.array([[0.67826139], [0.29380381], [0.90714982],
                [0.52835647], [0.4215251 ], [0.45017551], [0.92814219], [0.96677647],
                [0.85304703], [0.52351845], [0.19981397], [0.27417313], [0.60659855],
                [0.00533165], [0.10820313], [0.49978937], [0.34144279], [0.94630077]])
        self.assertEqual(image.image2vector(x_img).shape , y_img.shape)
        self.assertEqual(image.image2vector(x_img).any() , y_img.any())

if __name__ == '__main__':
    unittest.main()