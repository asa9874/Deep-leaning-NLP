import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 데이터 생성
X = np.array([[1], [2], [3], [4], [5]])  # 독립 변수
y = np.array([2, 4, 6, 8, 10])          # 종속 변수

# 모델 생성 및 학습
model = LinearRegression()
model.fit(X, y)

# 회귀 계수 출력
print(f"회귀 계수 (w): {model.coef_[0]}")
print(f"절편 (b): {model.intercept_}")

# 예측
new_data = np.array([[6]])
predicted = model.predict(new_data)
print(f"예측 값: {predicted[0]}")


# 시각화
plt.scatter(X, y, color='blue', label='data')  # 데이터 포인트
plt.plot(X, model.predict(X), color='red', label='line')  # 회귀선

# 예측 데이터 포인트 추가
plt.scatter(new_data, predicted, color='green', label='(6, 12)')

# 그래프 꾸미기
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()