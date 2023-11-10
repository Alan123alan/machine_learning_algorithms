import numpy as np

class LogisticRegression:
    def __init__(self, features, targets):
        self.X = np.array(features)
        self.m, self.n = np.shape(self.X)
        self.Y = np.array(targets)
        self.W = np.zeros(self.n)
        self.b = 0
    
    def train(self, iterations, learning_rate):
        for _ in range(iterations):
            linear_regression_model = self._model()
            Y_pred = self._sigmoid(linear_regression_model) 
            dw = (1/self.m)*np.dot(np.transpose(self.X),(Y_pred - self.Y))
            db = (1/self.m)*np.sum(Y_pred - self.Y)
            self.W -= learning_rate * dw
            self.b -= learning_rate * db

    def _model(self):
        return np.dot(self.X, self.W) + self.b

    def _sigmoid(self, Z):
        return 1/(1 + np.exp(-Z))
    
    def predict(self):
        linear_regression_model = np.dot(self.X, self.W) + self.b
        Y_predicted = self._sigmoid(linear_regression_model)
        return np.array([1 if i>0.5 else 0 for i in Y_predicted])
