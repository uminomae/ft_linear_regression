import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# サンプルデータの生成
np.random.seed(42)
n_samples = 300
n_features = 5
X = np.random.randn(n_samples, n_features)

# データの標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCAの実行
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 累積寄与率の計算
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# 結果の可視化
plt.figure(figsize=(10, 5))

# スクリープロット
plt.subplot(1, 2, 1)
plt.plot(range(1, n_features + 1), pca.explained_variance_ratio_, 'bo-')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')

# 累積寄与率プロット
plt.subplot(1, 2, 2)
plt.plot(range(1, n_features + 1), cumulative_variance_ratio, 'ro-')
plt.title('Cumulative Proportion of Variance Explained')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Proportion of Variance')
plt.axhline(y=0.8, color='g', linestyle='--')

plt.tight_layout()
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Cumulative explained variance ratio:", cumulative_variance_ratio)