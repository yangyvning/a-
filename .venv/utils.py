import pickle
import numpy as np

def load_batch(path):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    X = data[b'data'].astype(np.float32)
    y = np.array(data[b'labels'])
    return X, y

def load_cifar10():
    X, y = [], []
    for i in range(1, 6):
        X_batch, y_batch = load_batch(f"./cifar-10-batches-py/data_batch_{i}")
        X.append(X_batch)
        y.append(y_batch)
    X = np.vstack(X)
    y = np.hstack(y)

    # 分训练集 90% + 验证集 10%
    split = int(0.9 * len(X))
    return X[:split], y[:split], X[split:], y[split:]

def load_cifar10_test():
    X_test, y_test = load_batch("./cifar-10-batches-py/test_batch")
    return X_test, y_test

def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

def normalize(X):
    X = X / 255.0
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-7
    return (X - mean) / std