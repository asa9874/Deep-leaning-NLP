import numpy as np
import matplotlib.pyplot as plt

# 하이퍼볼릭 탄젠트 함수 정의
def tanh(x):
    return np.tanh(x)

# 입력값
x = np.linspace(-5, 5, 100)

# 그래프
plt.plot(x, tanh(x), label="tanh(x)")
plt.title("Hyperbolic Tangent Function (tanh)")
plt.xlabel("x")
plt.ylabel("tanh(x)")
plt.grid(True)
plt.legend()
plt.show()
