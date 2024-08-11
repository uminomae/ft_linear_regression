# 必要なライブラリの読み込み
import scipy.stats as st
from scipy.stats import multivariate_normal 
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# データの設定
x, y = np.mgrid[10:100:2, 10:100:2]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x 
pos[:, :, 1] = y

# 多次元正規分布
# それぞれの変数の平均と分散共分散行列を設定
rv = multivariate_normal([50, 50], [[100, 0], [0, 100]])

# 確率密度関数 
z = rv.pdf(pos)

# グラフの作成
fig = plt.figure(figsize=(10, 8), dpi=100)
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z)

# x,y,zラベルの設定など
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')

# z軸の表示目盛り単位を変更
ax.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))

# グラフの表示
plt.show()