import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. 데이터 생성 (성적과 통과 여부)
data = {
    "Score": [35, 50, 65, 70, 85, 40, 60, 75, 90, 45],
    "Pass": [0, 0, 0, 1, 1, 0, 1, 1, 1, 0],  # 1: 통과, 0: 불통과
}
df = pd.DataFrame(data)

X = df[["Score"]]  # 입력: 성적
y = df["Pass"]     # 출력: 통과 여부

# 2. 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 로지스틱 회귀 모델 정의 및 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. 예측 확률 계산
scores = np.linspace(30, 100, 300).reshape(-1, 1)  # 30점부터 100점까지 데이터 생성
probabilities = model.predict_proba(scores)[:, 1]  # 클래스 1(통과)의 확률

# 5. 시각화
plt.figure(figsize=(8, 6))

# 학습 데이터 산점도
plt.scatter(df["Score"], df["Pass"], color="blue", label="Training Data", zorder=5)

# 로지스틱 회귀 모델의 예측 확률 곡선
plt.plot(scores, probabilities, color="red", label="Logistic Regression Curve", linewidth=2)

# 그래프 꾸미기
plt.axhline(0.5, color="gray", linestyle="--", label="Decision Boundary (0.5)")
plt.title("Logistic Regression: Score vs Pass Probability")
plt.xlabel("Score")
plt.ylabel("Pass Probability")
plt.legend()
plt.grid(True)

# 그래프 출력
plt.show()
