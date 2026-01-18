import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])

np.random.seed(42)

W1 = np.random.randn(2, 2)
b1 = np.zeros((1, 2))

W2 = np.random.randn(2, 1)
b2 = np.zeros((1, 1))

learning_rate = 0.1
epochs = 200
losses = []

for _ in range(epochs):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    y_pred = sigmoid(z2)

    loss = np.mean((y - y_pred) ** 2)
    losses.append(loss)

    d_y_pred = (y_pred - y) * sigmoid_derivative(y_pred)

    dW2 = np.dot(a1.T, d_y_pred)
    db2 = np.sum(d_y_pred, axis=0, keepdims=True)

    d_hidden = np.dot(d_y_pred, W2.T) * sigmoid_derivative(a1)

    dW1 = np.dot(X.T, d_hidden)
    db1 = np.sum(d_hidden, axis=0, keepdims=True)

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("loss_plot.png")
plt.show()

print(np.round(y_pred, 3))
