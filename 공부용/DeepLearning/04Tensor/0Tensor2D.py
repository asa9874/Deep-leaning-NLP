import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 2D 텐서 (행렬) 히트맵 시각화
tensor_2d = np.array([[1, 2], [3, 4]])
sns.heatmap(tensor_2d, annot=True, fmt="d", cmap="Blues")
plt.title("2D Tensor")
plt.show()
