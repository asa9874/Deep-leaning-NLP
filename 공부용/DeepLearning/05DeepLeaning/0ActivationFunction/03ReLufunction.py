import numpy as np
import matplotlib.pyplot as plt

# ReLU 함수 정의
def relu(x):
    return np.maximum(0, x)

# 입력값
x = np.linspace(-5, 5, 100)

# 그래프
plt.plot(x, relu(x), label="ReLU(x)")
plt.title("ReLU Function")
plt.xlabel("x")
plt.ylabel("ReLU(x)")
plt.grid(True)
plt.legend()
plt.show()
