import numpy as np
import matplotlib.pyplot as plt

# サンプルデータ
mileages = np.array([100000, 150000, 200000, 250000, 300000])
mean = np.mean(mileages)
std_dev = np.std(mileages)

# 正規化の各ステップ
step1 = mileages - mean
step2 = step1 / std_dev

# プロット
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

# 元のデータ
ax1.scatter(range(len(mileages)), mileages)
ax1.set_title("Original Data")
ax1.axhline(y=mean, color='r', linestyle='--', label='Mean')
ax1.legend()

# 平均を引いた後
ax2.scatter(range(len(step1)), step1)
ax2.set_title("After Subtracting Mean")
ax2.axhline(y=0, color='r', linestyle='--', label='New Mean (0)')
ax2.legend()

# 標準偏差で割った後
ax3.scatter(range(len(step2)), step2)
ax3.set_title("After Dividing by Standard Deviation")
ax3.axhline(y=0, color='r', linestyle='--', label='Mean (0)')
ax3.axhline(y=1, color='g', linestyle='--', label='1 Std Dev')
ax3.axhline(y=-1, color='g', linestyle='--', label='-1 Std Dev')
ax3.legend()

plt.tight_layout()
plt.show()