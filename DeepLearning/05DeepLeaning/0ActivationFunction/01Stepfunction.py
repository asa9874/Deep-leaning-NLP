import numpy as np
import matplotlib.pyplot as plt

# 계단 함수 정의
def step_function(x):
    return np.where(x > 0, 1, 0)

# 입력 값 범위 생성
x = np.linspace(-10, 10, 100)  # -10에서 10까지 100개의 점
y = step_function(x)  # 계단 함수 적용

# 시각화
plt.figure(figsize=(8, 4))
plt.plot(x, y, label="Step Function", color="blue")
plt.title("Step Function Visualization")
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # x축
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')  # y축
plt.xlabel("Input (x)")
plt.ylabel("Output (y)")
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend()
plt.show()
