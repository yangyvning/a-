import numpy as np
import matplotlib.pyplot as plt
from model import NeuralNet

# 加载已训练模型参数
params = np.load("saved_model.npz")
W1 = params["W1"]  # W1.shape = (3072, hidden_size)

# 可视化前 10 个神经元的权重
hidden_num_to_show = 10
plt.figure(figsize=(15, 4))

for i in range(hidden_num_to_show):
    # 取出第 i 个神经元的权重列，reshape 为原图形状
    w = W1[:, i].reshape(3, 32, 32)  # shape: (3, 32, 32)
    w = w.transpose(1, 2, 0)  # shape: (32, 32, 3)

    # 归一化权重到 0-1 区间，便于显示
    w_min, w_max = np.min(w), np.max(w)
    w_norm = (w - w_min) / (w_max - w_min + 1e-7)

    plt.subplot(2, 5, i + 1)
    plt.imshow(w_norm)
    plt.axis('off')
    plt.title(f"Filter {i + 1}")

plt.suptitle("Visualization of the weights of the first layer hidden neurons (top 10)")
plt.tight_layout()
plt.savefig("weights_visual.png")
plt.show()
