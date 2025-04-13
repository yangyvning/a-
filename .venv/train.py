import numpy as np
import matplotlib.pyplot as plt
from model import NeuralNet
from utils import load_cifar10, one_hot, normalize

def train(hidden_size=512, epochs=100, lr=0.1, reg=1e-5, batch_size=200, dropout=0.5):
    X_train, y_train, X_val, y_val = load_cifar10()
    X_train, X_val = normalize(X_train), normalize(X_val)
    y_train_onehot = one_hot(y_train, 10)
    y_val_onehot = one_hot(y_val, 10)

    model = NeuralNet(3072, hidden_size, 10, dropout_rate=dropout)

    best_acc = 0
    best_params = {}
    train_losses, val_losses, val_accuracies = [], [], []
    patience, wait = 10, 0

    for epoch in range(epochs):
        indices = np.random.permutation(len(X_train))
        X_train, y_train_onehot = X_train[indices], y_train_onehot[indices]

        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train_onehot[i:i+batch_size]

            loss, grads = model.loss(X_batch, y_batch, reg)
            model.W1 -= lr * grads["W1"]
            model.b1 -= lr * grads["b1"]
            model.W2 -= lr * grads["W2"]
            model.b2 -= lr * grads["b2"]

        # 验证
        train_loss, _ = model.loss(X_train, y_train_onehot, reg, training=False)
        val_loss, _ = model.loss(X_val, y_val_onehot, reg, training=False)
        y_val_pred = model.predict(X_val)
        val_acc = np.mean(y_val_pred == y_val)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            wait = 0
            best_params = {
                "W1": model.W1.copy(),
                "b1": model.b1.copy(),
                "W2": model.W2.copy(),
                "b2": model.b2.copy()
            }
            np.savez("saved_model.npz", **best_params)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

        lr *= 0.95  # 学习率衰减

    # 可视化训练集 loss 曲线
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.legend()
    plt.title("Training Loss")
    plt.savefig("train_loss.png")
    plt.clf()



if __name__ == "__main__":
    train()