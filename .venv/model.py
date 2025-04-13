import numpy as np

class NeuralNet:
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)
        self.dropout_rate = dropout_rate
        self.dropout_mask = None

    def relu(self, x):
        return np.maximum(0, x)

    def relu_grad(self, x):
        return x > 0

    def softmax(self, x):
        x -= np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def loss(self, X, y, reg=0.0, training=True):
        N = X.shape[0]

        # Forward
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)

        if training and self.dropout_rate > 0:
            self.dropout_mask = (np.random.rand(*a1.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
            a1 *= self.dropout_mask

        scores = a1 @ self.W2 + self.b2
        probs = self.softmax(scores)

        correct_logprobs = -np.log(probs[np.arange(N), np.argmax(y, axis=1)] + 1e-7)
        data_loss = np.mean(correct_logprobs)
        reg_loss = 0.5 * reg * (np.sum(self.W1**2) + np.sum(self.W2**2))
        loss = data_loss + reg_loss

        # Backward
        dscores = probs - y
        dscores /= N

        grads = {}
        grads["W2"] = a1.T @ dscores + reg * self.W2
        grads["b2"] = np.sum(dscores, axis=0)

        da1 = dscores @ self.W2.T
        if training and self.dropout_rate > 0:
            da1 *= self.dropout_mask

        dz1 = da1 * self.relu_grad(z1)
        grads["W1"] = X.T @ dz1 + reg * self.W1
        grads["b1"] = np.sum(dz1, axis=0)

        return loss, grads

    def predict(self, X):
        a1 = self.relu(X @ self.W1 + self.b1)
        if self.dropout_rate > 0:
            a1 *= (1.0 - self.dropout_rate)
        scores = a1 @ self.W2 + self.b2
        return np.argmax(scores, axis=1)
