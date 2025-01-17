import numpy as np

import matplotlib.pyplot as plt
# 소프트맥스 함수 정의
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # overflow를 방지하기 위해 최대값을 빼줌
    return exp_x / np.sum(exp_x)

# 여러 가지 입력값에 대해 소프트맥스 함수 출력
x = np.linspace(-5, 5, 100)
y = softmax(x)

plt.plot(x, y, label="Softmax(x)")
plt.title("Softmax Function")
plt.xlabel("x")
plt.ylabel("Softmax(x)")
plt.grid(True)
plt.legend()
plt.show()
