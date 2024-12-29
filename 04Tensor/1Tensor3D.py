from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# 3D 텐서 점 그래프
tensor_3d = np.random.rand(10, 3)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tensor_3d[:, 0], tensor_3d[:, 1], tensor_3d[:, 2])
ax.set_title("3D Tensor")
plt.show()
