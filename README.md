本项目手工实现三层神经网络分类器，用于对 CIFAR-10 图像进行分类。 使用说明

安装依赖 pip install numpy Pip install matplotlib

数据准备 从 https://www.cs.toronto.edu/~kriz/cifar.html 下载并解压数据集。

训练模型 python train.py 默认参数下将在训练过程中保存最优模型至 saved_model.npz

测试模型 python test.py 加载 saved_model.npz 后评估测试集准确率。

参数可视化 python visualize_weights.py 生成 weights_visual.png 展示第一层神经元权重图案。

模型结构 输入层：3072维（32×32×3） 隐藏层：ReLU + Dropout 输出层：Softmax（10类） 训练特性 动态学习率衰减 Dropout 正则 L2 正则 Early Stopping

项目结构 ├── train.py ├── test.py ├── model.py ├── utils.py ├── visualize_weights.py ├── saved_model.npz ├── loss_curve.png └── filters.png# -# a-
