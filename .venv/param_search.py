# param_search.py
from train import train

hidden_sizes = [64, 128, 256,512,1024]
learning_rates = [1e-1, 1e-2, 1e-3,1e-4,1e-5]
regs = [0.0, 1e-3, 1e-2]

for h in hidden_sizes:
    for lr in learning_rates:
        for reg in regs:
            print(f"\nTesting: hidden={h}, lr={lr}, reg={reg}")
            train(hidden_size=h, lr=lr, reg=reg, epochs=10)
