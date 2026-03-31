""" 

- 1. This project it's a Perceptron (single-layer neural network, the loss function are different but fuck it lol) implementation from scratch in Python. The goal of this project is to understand the inner workings of a Perceptron and how it can be used for classification tasks.
- 2. All the dataset are randomly generated and the model is not trained, it's just a skeleton of a Perceptron implementation. The user can fill in the logic for the forward pass, loss function, backward pass and optimization algorithm.

"""


import numpy as np

# Get the data 
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler 

# 1. Preparação dos dados
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target.reshape(-1, 1)

# NORMALIZAÇÃO: Essencial para redes neurais
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]



class Model:
    def __init__(self, input_size):
        # Define the bias and weights
        self.weights = np.random.randn(input_size, 1) * 0.01
        self.bias = 0.0

    # Activation function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    # Loss
    def loss(self, y_hat, y):
        m = y.shape[0]
        loss = -1/m * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return loss
    
    # Backprop
    def optim(self, X, y, y_hat, learning_rate):
        m = y.shape[0]
        dw = 1/m * np.dot(X.T, (y_hat - y))
        db = 1/m * np.sum(y_hat - y)
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db

    def forward_pass(self, X):
        # Fordward pass
        z = np.dot(X, self.weights) + self.bias

        # Activation function
        y_hat = self.sigmoid(z)
        return y_hat


model = Model(input_size=dataset.data.shape[1])

for epoch in range(100):

    # Forward pass
    y_hat = model.forward_pass(X_train)

    # Loss calculation
    current_loss = model.loss(y_hat, y_train)

    # Optimization
    model.optim(X_train, y_train, y_hat, learning_rate=0.03) 

    # Print the loss and epoch every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {current_loss:.4f}")

# Test accuracy on the training set
pred_classes = (y_hat > 0.5).astype(int)
accuracy = np.mean(pred_classes == y_train)
print(f"\nTest accuracy: {accuracy * 100:.2f}%")

