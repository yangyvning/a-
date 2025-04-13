import numpy as np
import matplotlib.pyplot as plt
from model import NeuralNet
from utils import load_cifar10_test, one_hot, normalize

def test():
    X_test, y_test = load_cifar10_test()
    X_test = normalize(X_test)
    y_test_onehot = one_hot(y_test, 10)

    # 加载训练好的模型
    data = np.load("saved_model.npz")
    model = NeuralNet(3072, 512, 10, dropout_rate=0.0)
    model.W1, model.b1 = data["W1"], data["b1"]
    model.W2, model.b2 = data["W2"], data["b2"]

    # 预测
    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print(f"Overall Test Accuracy: {acc:.4f}")

    # 计算 loss 和 accuracy 曲线（用 batch 测试）
    batch_size = 500
    test_losses, test_accuracies = [], []

    for i in range(0, len(X_test), batch_size):
        X_batch = X_test[i:i+batch_size]
        y_batch = y_test_onehot[i:i+batch_size]
        true_labels = y_test[i:i+batch_size]

        loss, _ = model.loss(X_batch, y_batch, reg=0.0, training=False)
        preds = model.predict(X_batch)
        accuracy = np.mean(preds == true_labels)

        test_losses.append(loss)
        test_accuracies.append(accuracy)

    # 可视化
    plt.plot(test_losses, label="Test Loss per Batch")
    plt.plot(test_accuracies, label="Test Accuracy per Batch")
    plt.xlabel("Batch")
    plt.ylabel("Metric Value")
    plt.title("Test Loss & Accuracy")
    plt.legend()
    plt.savefig("test_metrics.png")
    print("Saved: test_metrics.png")

if __name__ == "__main__":
    test()
