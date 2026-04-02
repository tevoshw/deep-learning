import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler


# Get the data
dataset = load_diabetes()
X = dataset.data
y = dataset.target.reshape(-1, 1)

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)


# Split
X_train, X_test = X[:350], X[350:]
y_train, y_test = y[:350], y[350:]



"""
So have 442 samples, 10 features (weigths)


And we gonna create

INPUT LAYER |       HIDDEN LAYER |          OUTPUT LAYER
  10 NEURONS        5 NEURONS                  1 NEURON
(THE FEATURES)
"""


size_features = X.shape[1]


class MultiLayerPerceptron:
    def __init__(self, size_features):
        self.neurons = 5
        self.weights_hidden = np.random.randn(size_features, self.neurons) * np.sqrt(1 / size_features)
        self.bias_hidden = np.zeros((1, self.neurons))

        self.weights_output = np.random.randn(self.neurons, 1) * np.sqrt(1 / self.neurons)
        self.bias_output = np.zeros((1, 1))

    def ReLu(self, Z):
        return np.maximum(0, Z)
    
    def Loss(self, y_hat, y_true):
        loss = np.mean((y_hat - y_true) ** 2)
        return loss
 
    def Forward_pass(self, X_train):
        self.z1 = np.dot(X_train, self.weights_hidden) + self.bias_hidden
        self.h1 = self.ReLu(self.z1)
        y_hat = np.dot(self.h1, self.weights_output) + self.bias_output
        return y_hat

    def Optimize(self, y_hat, y_true, X_train):
        m = X_train.shape[0]
        lr = 0.03

        error_out = y_hat - y_true 

        dW_OUTPUT = np.dot(self.h1.T, error_out) / m
        db_OUTPUT = np.sum(error_out, axis=0, keepdims=True) / m

        error_hidden = np.dot(error_out, self.weights_output.T) 
        error_hidden[self.z1 <= 0] = 0

        dW_HIDDEN = np.dot(X_train.T, error_hidden) / m
        db_HIDDEN = np.sum(error_hidden, axis=0, keepdims=True) / m

        self.weights_output -= lr * dW_OUTPUT
        self.bias_output    -= lr * db_OUTPUT
        self.weights_hidden -= lr * dW_HIDDEN
        self.bias_hidden    -= lr * db_HIDDEN

epochs = 150
model = MultiLayerPerceptron(size_features)

for x in range(epochs+ 1):
    y_hat = model.Forward_pass(X_train)
    loss = model.Loss(y_hat, y_train)
    optim = model.Optimize(y_hat, y_train, X_train)

    if x % 10 == 0:
        print(f'Epoch: {x}, loss: {loss:.4f}')


y_pred_test = model.Forward_pass(X_test)
test_loss = np.mean((y_pred_test - y_test)**2)

print("Test loss:", test_loss)

