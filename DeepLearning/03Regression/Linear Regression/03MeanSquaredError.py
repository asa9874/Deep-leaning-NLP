import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 데이터 생성 (오차 포함)
np.random.seed(42)  # 랜덤 시드 고정
X = np.array([[1], [2], [3], [4], [5]])  # 독립 변수
y = 2 * X.flatten() + 1 + np.random.normal(0, 1, size=X.shape[0])  # y = 2x + 1 + N(0, 1)

# 모델 생성 및 학습
model = LinearRegression()
model.fit(X, y)

# 회귀 계수 및 절편 확인
print(f"회귀 계수 (w): {model.coef_[0]}")
print(f"절편 (b): {model.intercept_}")

# 예측 값 확인
new_data = np.array([[6]])
predicted = model.predict(new_data)
print(f"예측 값: {predicted[0]}")

#MSE
y_pred = model.predict(X) #모델이 예측한 각x 에 대한 y값
mse = np.mean((y - y_pred) ** 2)
print("MSE:",mse)

# 시각화
plt.scatter(X, y, color='blue')  # 데이터 포인트
plt.plot(X, model.predict(X), color='red')  # 회귀선
plt.scatter(new_data, predicted, color='green', label=f'(6, {predicted[0]:.2f})')

# 그래프 꾸미기
plt.xlabel("(X)")
plt.ylabel("(y)")
plt.legend()
plt.grid()
plt.show()
