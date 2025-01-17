# 필요한 라이브러리 import
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 임의 데이터 생성
np.random.seed(42)

# 100개의 샘플, 각 샘플은 3개의 특성(feature)을 가짐
X = np.random.randn(100, 3)

# 3개의 클래스를 가진 레이블 생성 (0, 1, 2)
y = np.random.choice([0, 1, 2], size=100)

# 훈련 세트와 테스트 세트로 데이터 나누기
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# 소프트맥스 회귀 모델 학습
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"모델의 정확도: {accuracy * 100:.2f}%")

# 예측 결과 출력 (테스트 데이터에 대한 예측 클래스)
print(f"예측 클래스: {y_pred[:10]}")

# 시각화: X_train의 첫 2개의 특성만 사용하여 2D 플로팅
plt.figure(figsize=(8, 6))

# 각 클래스별로 다른 색상을 지정하여 데이터 시각화
for class_label in np.unique(y_train):
    plt.scatter(X_train[y_train == class_label, 0], X_train[y_train == class_label, 1],
                label=f'Class {class_label}', alpha=0.6)

# 모델의 예측 경계 시각화
h = .02  # 간격 설정
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())])
Z = Z.reshape(xx.shape)

# 결정 경계를 색으로 표시
plt.contourf(xx, yy, Z, alpha=0.3)
plt.title('Softmax Regression Decision Boundaries')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='best')
plt.show()
