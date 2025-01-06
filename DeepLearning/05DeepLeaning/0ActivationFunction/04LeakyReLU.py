import numpy as np
import matplotlib.pyplot as plt

# Leaky ReLU 함수 정의
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# 입력값
x = np.linspace(-5, 5, 100)

# 그래프
plt.plot(x, leaky_relu(x), label="Leaky ReLU(x)", color='green')
plt.title("Leaky ReLU Function")
plt.xlabel("x")
plt.ylabel("Leaky ReLU(x)")
plt.grid(True)
plt.legend()
plt.show()
