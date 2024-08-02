import csv
import json
# グラフ描画のためのライブラリ（pltというエイリアスで使用）
import matplotlib.pyplot as plt
# Pythonによる科学技術計算の基礎パッケージ（npというエイリアスで使用）
# 参考:【NumPy -】 <https://numpy.org/ja/>
import numpy as np

def estimate_price(mileage, theta0, theta1):
	return theta0 + (theta1 * mileage)

def train_model(data, alpha, num_iterations):
	"""
	勾配降下法
	∂MSE/∂a = (-2/n) * Σ(yi - (axi + b)) * xi  
	∂MSE/∂b = (-2/n) * Σ(yi - (axi + b))  
	
	MSE: 平均二乗誤差（Mean Squared Error）
	a: 傾き（theta1に相当）
	b: 切片（theta0に相当）
	n: データポイントの数
	xi: 各データポイントの入力値（走行距離）
	yi: 各データポイントの実際の出力値（価格）
	alpha: 学習率
	"""
	b, a = 0, 0
	n = len(data)

	# データからmileage（走行距離）とprice（価格）を抽出し、NumPy配列に変換
	# Convert lists to numpy arrays
	# x = 走行距離
	# y = 価格
	x = np.array([row[0] for row in data])  
	y = np.array([row[1] for row in data])  

	# データの平均（mean）と標準偏差（std）を計算
	x_mean, x_std = np.mean(x), np.std(x)
	y_mean, y_std = np.mean(y), np.std(y)

	# 正規化 = （データ - 平均）/ 標準偏差
	# - 標準偏差で割ることの意味：
	#   - 正規化されたデータの約68%が-1と1の間に、約95%が-2と2の間に収まる（正規分布の特性）。
	x_norm = (x - x_mean) / x_std
	y_norm = (y - y_mean) / y_std

	# num_iterations回だけループを実行
	for _ in range(num_iterations):
		# 予測値の計算: 線形回帰の基本形　y = ax + b の正規化バージョン
		y_pred = b + a * x_norm

		# 勾配の計算（正規化されたデータを使用）: 平均二乗誤差（MSE）の a と b に関する偏導関数
		#  ∂MSE/∂a = (-2/n) * Σ(yi - (θ1*xi + θ0)) * xi 
		#  ∂MSE/∂θ0 = (-2/n) * Σ(yi - (θ1*xi + θ0)) 
		# (-2/n) は定数項
		# n = データ点の数
		# Σ(y_norm - y_pred) = 実際の値と予測値の差（誤差）の合計
		grad_a = (-2/n) * np.sum((y_norm - y_pred) * x_norm)
		grad_b = (-2/n) * np.sum(y_norm - y_pred)

		# パラメータの更新
		# a = a - η * (∂MSE/∂a) 
		# b = b - η * (∂MSE/∂b)
		a = a - alpha * grad_a
		b = b - alpha * grad_b

	# 非正規化: 正規化されたパラメータを元のスケールに戻す
	# 1. 基本的な関係式:
	#    正規化前: y = a_original * x + b_original
	#    正規化後: y_norm = a * x_norm + b
	# 2. 正規化の定義:
	#    x_norm = (x - x_mean) / x_std
	#    y_norm = (y - y_mean) / y_std
	# 3. 正規化された式を元の変数で表現:
	#    y_normとx_normを定義で書き直すと: y_norm = (y - y_mean) / y_std = a * x_norm + b = a * (x - x_mean) / x_std + b
	#    (y - y_mean) / y_std = a * (x - x_mean) / x_std + b
	# 4. 式変形のステップ:
	#    a) 両辺に y_std をかける:
	#       y - y_mean = a * (y_std / x_std) * (x - x_mean) + b * y_std
	#    b) 右辺の括弧を展開:
	#       y - y_mean = a * (y_std / x_std) * x - a * (y_std / x_std) * x_mean + b * y_std
	#    c) 両辺に y_mean を足す:
	#       y = a * (y_std / x_std) * x - a * (y_std / x_std) * x_mean + b * y_std + y_mean
	# 5. 正規化前の関係式と比較:
	#    y = a_original * x + b_original
	#
	#    したがって:
	#    a_original = a * (y_std / x_std)
	#    b_original = b * y_std + y_mean - a * (y_std / x_std) * x_mean
	# 6. 非正規化の実行:
	a = a * (y_std / x_std)
	b = (b * y_std) + y_mean - (a * x_mean)

	return b, a

def main():
	data = []
	# 読み込みモードで開く
	with open('data.csv', 'r') as f:
		# CSVファイルを読み込むためのreaderオブジェクトを作成
		csv_reader = csv.reader(f)
		# データの1行目を無視　ヘッダーをスキップ
		next(csv_reader) 
		# 各行をループで処理し、xとyの値を浮動小数点数に変換してタプルとしてdataリストに追加
		for row in csv_reader:
			x, y = float(row[0]), float(row[1])
			data.append((x, y))

	# 学習率
	alpha = 0.01
	# 反復回数
	num_iterations = 1000
	# モデルを訓練し、結果（切片bと傾きa）を取得
	# https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files
	b, a = train_model(data, alpha, num_iterations)

	# 書き込みモードで開く
	with open('model_parameters.json', 'w') as f:
		# 学習したパラメータ（bとa）をJSON形式でファイルに保存
		json.dump({'b': b, 'a': a}, f)

	# 訓練完了のメッセージ
	print(f"Training complete. b (intercept): {b}, a (slope): {a}")

	# データポイントを青色の散布図としてプロット
	plt.scatter([row[0] for row in data], [row[1] for row in data], color='blue', label='Data points')
	# 線形回帰モデルを赤い線としてプロット
	# 最小値から最大値までの範囲で線を引く
	plt.plot([min(row[0] for row in data), max(row[0] for row in data)], 
			[estimate_price(min(row[0] for row in data), b, a), 
			estimate_price(max(row[0] for row in data), b, a)], 
			color='red', label='Linear regression')
	# グラフにx軸ラベル、y軸ラベル、タイトル、凡例を追加
	plt.xlabel('Mileage (x)')
	plt.ylabel('Price (y)')
	plt.title('Car Price vs Mileage')
	plt.legend()
	# 5秒間グラフを表示　（ブロッキングせずに）して閉じる
	plt.show(block=False)
	plt.pause(5)
	plt.close()

if __name__ == "__main__":
	main()