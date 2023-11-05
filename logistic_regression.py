import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

# class logisticRegression:

#   def __init__(self,lr=0.001,n_iters=1000):
#     self.lr = lr
#     self.n_iters = n_iters
#     self.weights = None
#     self.bias = None

#   def fit(self,X,y):
#     #init parameters
#     n_samples, n_features = X.shape
#     self.weights = np.zeros(n_features)
#     self.bias = 0

#     #gradient descent
#     for _ in range(self.n_iters):
#       linear_model = np.dot(X,self.weights) + self.bias
#       y_predicted = self._sigmoid(linear_model)

#       dw = (1/n_samples) * np.dot(X.T,(y_predicted-y))
#       db = (1/n_samples) * np.sum(y_predicted-y)

#       self.weights -= self.lr *dw
#       self.bias -= self.lr * db 

#   def predict(self,X):
#     linear_model = np.dot(X,self.weights) + self.bias
#     y_predicted = self._sigmoid(linear_model)
#     y_predicted_cls = [1 if i>0.5 else 0 for i in y_predicted]
#     return y_predicted_cls
  
#   def _sigmoid(self,x):
#     return(1/(1+np.exp(-x)))

def sigmoid(z):
  return 1/(1+(np.exp(-z)))

def training_loop(iterations, learning_rate, m, w, b, features, targets):
    for _ in range(iterations):
        y_pred = sigmoid(np.dot(features, w) + b)
        dw = (1/m)*np.dot(features.T,(y_pred - targets))
        db = (1/m)*np.sum(y_pred - targets)
        w -= learning_rate * dw
        b -= learning_rate * db
    return w, b

def accuracy(y_true,y_pred):
  accuracy = np.sum(y_true == y_pred)/len(y_true)
  return accuracy

#Test Logistic regression in breast_cancer
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
m, n = X_train.shape
W = np.zeros(n)
b = 0
print(W, b)
trained_weights, trained_bias = training_loop(1000, 0.0001, m, W, b, X_train, y_train)
print(trained_weights, trained_bias)
print(np.array(X_train))
print(np.array(y_train))

# regressor = logisticRegression(lr=0.0001,n_iters=5000)

# regressor.fit(X_train, y_train)

# predictions = regressor.predict(X_test)
linear_model = np.dot(X_test,W) + b
y_predicted = sigmoid(linear_model)
y_predicted_cls = [1 if i>0.5 else 0 for i in y_predicted]

print("Accuracy: ",accuracy(y_test, y_predicted_cls))
print(y_test, y_predicted_cls)