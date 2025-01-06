import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 데이터 생성
X = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])  # 독립 변수 
y = np.array([5, 10, 15, 20])                  # 종속 변수 

# 모델 생성 및 학습
model = LinearRegression()
model.fit(X, y)

# 회귀 계수 및 절편 확인
print(f"회귀 계수 (w1, w2): {model.coef_}")
print(f"절편 (b): {model.intercept_}")

# 예측
new_data = np.array([[5, 10]])
predicted = model.predict(new_data)
print(f"예측 값: {predicted[0]}")



# 시각화 
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 기존 데이터 포인트 그리기
ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='data')

# 회귀 평면 생성
x1_range = np.linspace(min(X[:, 0]), max(X[:, 0]), 10)  # X1 범위
x2_range = np.linspace(min(X[:, 1]), max(X[:, 1]), 10)  # X2 범위
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)      # 그리드 생성
y_grid = (model.coef_[0] * x1_grid + model.coef_[1] * x2_grid + model.intercept_)  # 회귀 평면

# 회귀 평면 그리기
ax.plot_surface(x1_grid, x2_grid, y_grid, color='red', alpha=0.5)

# 예측 포인트 추가
ax.scatter(new_data[:, 0], new_data[:, 1], predicted, color='green', s=100, label='(5, 10, 25)')

# 축 레이블 설정
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')
ax.set_title("MultipleLinear")
ax.legend()

plt.show()