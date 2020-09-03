import unittest
import numpy as np
import logisticregression as lr

model = lr.LogisticRegression()

w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])

class TestLogisticRegressionMethods(unittest.TestCase):

    def test_propagate(self):
        grads, cost = model.propagate(w, b, X, Y)
        self.assertEqual(grads["db"],  0.001455578136784208)
        self.assertEqual(cost, 5.801545319394553)
        self.assertEqual(grads["dw"].any(), np.array([[0.99845601], [2.39507239]]).any())
    
    def test_optimize(self):
        params, grads, costs = model.optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009)
        self.assertEqual(params["w"].any(), np.array([[0.99845601], [2.39507239]]).any())
        self.assertEqual(params["b"], 1.9253598300845747)
        self.assertEqual(grads['dw'].any(), np.array([[0.67752042],[1.41625495]]).any())
        self.assertEqual(grads['db'], 0.21919450454067657)
        self.assertEqual(len(costs), 1)
    
    def test_predict(self):
        w = np.array([[0.1124579],[0.23106775]])
        b = -0.3
        X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
        pred = model.predict_(w, b, X)
        self.assertEqual(pred.any(), np.array([[1,1,0]]).any())

if __name__ == '__main__':
    unittest.main()

