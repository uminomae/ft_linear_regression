# ft_linear_regression

## 概要

Pythonによる機械学習の初歩：線形回帰を勾配降下法で反復的に近似解を求める  

##　参考書

- 【東京大学のデータサイエンティスト育成講座 | 塚本 邦尊, 山田 典一, 大澤 文孝, 中山 浩太郎（監修）, 松尾 豊（監修）
- 実践的データ基盤への処方箋〜 ビジネス価値創出のためのデータ・システム・ヒトのノウハウ ゆずたそ (著), 渡部 徹太郎 (著), 伊藤 徹郎 (著)

## 以下、後で詳細に調べるための個人メモ

### 線形回帰問題を解くための3つの主要な手法（最小二乗法、正規方程式、勾配降下法）の特徴

| 特性 | 最小二乗法 | 正規方程式 | 勾配降下法 |
|------|------------|------------|------------|
| **定義** | 残差平方和を最小化する方法 | 最小二乗法を行列形式で表現し、解く方法 | 損失関数の勾配を用いて反復的に解を求める方法 |
| **アプローチ** | 解析的 | 解析的 | 数値的（反復的） |
| **計算式** | Σ(yi - (axi + b))² を最小化 | θ = (X^T X)^(-1) X^T y | θ = θ - α * ∇J(θ) （反復的に適用） |
| **計算量** | O(n^2) | O(n^3) | O(kn^2) （kは反復回数） |
| **データ規模への適合性** | 小～中規模 | 小～中規模 | 大規模 |
| **メモリ要件** | 中 | 高 | 低（バッチ処理可能） |
| **収束性** | 保証される | 保証される | 学習率に依存 |
| **精度** | 高 | 高 | 近似解（反復回数に依存） |
| **柔軟性** | 線形問題に限定 | 線形問題に限定 | 様々な最適化問題に適用可能 |
| **特徴** | ・直感的で理解しやすい<br>・小規模データに適している | ・一度の計算で厳密解を得られる<br>・逆行列計算が必要 | ・大規模データに適している<br>・ハイパーパラメータの調整が必要 |
| **課題** | ・大規模データでは計算コストが高い | ・特徴量が多い場合に計算コストが非常に高くなる<br>・多重共線性に弱い | ・局所最適解に陥る可能性がある<br>・収束に時間がかかる場合がある |

### 微分の基本概念：

- 目的: 「傾き」を求めるため
  - 微分は関数の「変化率」

線形関数 y = ax + b の場合、その導関数（微分した結果）は常に定数 a 
つまり、dy/dx = a 

### 勾配降下法における傾きの計算：

- 実際のデータポイントと予測値の差（誤差）を最小化することが目標
  - この誤差を表す関数を「損失関数」と呼ぶ
  - 一般的に使われる損失関数の一つが「平均二乗誤差（MSE）」

#### MSEの定義:

n個のデータポイント (xi, yi) に対して、  
MSE = (1/n) * Σ(yi - (axi + b))²  
ここで、Σは i=1 から n までの和を表します。 
この数式は、実際の値と予測値の差を二乗して平均を取っていることを表しています。  

- yi は実際のデータポイントのy値
- (axi + b) は線形関数による予測値
- (yi - (axi + b)) は実際の値と予測値の差（誤差）
- (yi - (axi + b))² は誤差を二乗した値
- (1/n) * をかけることで、合計した二乗誤差の平均を取ります

- 二乗を使う理由
  - 正の誤差と負の誤差を同等に扱えます
  - 大きな誤差により重みを置くことができます
  - 数学的に扱いやすくなります（微分が簡単になる）

### 偏微分の概念：

偏微分は、複数の変数を持つ関数において、一つの変数に注目してその変数に関する微分を行う

- 例えば、z = f(x, y) という2変数関数があるとき：  

xに関する偏微分：∂z/∂x は、yを定数として扱い、xだけを変数として微分します。  
yに関する偏微分：∂z/∂y は、xを定数として扱い、yだけを変数として微分します。  

- MSEの場合、aとbの2つのパラメータがあるので、それぞれに関して偏微分を行います：

∂MSE/∂a：bを固定して、aだけを変化させたときのMSEの変化率  
∂MSE/∂b：aを固定して、bだけを変化させたときのMSEの変化率  

### 傾きの計算：

- 勾配降下法では、この損失関数に対するパラメータ a と b の偏微分を計算  
  - 損失関数が a と b の変化に対してどのように変化するかを表す  

∂MSE/∂a = (-2/n) * Σ(yi - (axi + b)) * xi  
∂MSE/∂b = (-2/n) * Σ(yi - (axi + b))  

- 式の導出
MSE = (1/n) * Σ(yi - (axi + b))²  
これを a と b について偏微分  

MSEを a で偏微分すると：  
∂MSE/∂a = ∂/∂a [(1/n) * Σ(yi - (axi + b))²]  
= (1/n) * Σ∂/∂a [(yi - (axi + b))²]  
= (1/n) * Σ[2(yi - (axi + b)) * (-xi)]  （チェーンルールを適用）  
= (-2/n) * Σ[(yi - (axi + b)) * xi]  

b で偏微分すると：  
∂MSE/∂b = ∂/∂b [(1/n) * Σ(yi - (axi + b))²]  
= (1/n) * Σ∂/∂b [(yi - (axi + b))²]  
= (1/n) * Σ[2(yi - (axi + b)) * (-1)]  （チェーンルールを適用）  
= (-2/n) * Σ(yi - (axi + b))  

### パラメータの更新：  

- 計算された傾きを使って、a と b を以下のように更新  

a = a - η * (∂MSE/∂a)  
b = b - η * (∂MSE/∂b)  

ここで、η（イータ）は学習率と呼ばれる小さな正の数で、一度にどれだけパラメータを更新するかを制御します。  
このプロセスを繰り返すことで、徐々に最適な a と b の値に近づいていきます  

これらの偏微分を計算することで、aとbをそれぞれどう変更すればMSEが小さくなるかがわかります。

### 正規化（Normalization）

- 定義と目的: データを共通のスケールに変換するプロセス。
  - 特徴量間のスケールの違いによる影響を減らす
  - 勾配降下法の収束を早め、安定させる
  - モデルの学習を容易にし、精度を向上
- 計算: 正規化 = （データ - 平均）/ 標準偏差
- 結果: データの平均が0、標準偏差が1になる
- 重要性:
  - 収束速度の向上
  - 数値的安定性の確保
  - 特徴量の重要度の均等化
- 非正規化（逆変換）:学習後のモデルを元のスケールで使用するために必要。
- 式:
  - a_original = a * (y_std / x_std)  
  - b_original = (b * y_std) + y_mean - (a * y_std / x_std) * x_mean  
- 注意点: テストデータも同じ平均と標準偏差で正規化

下記の図は標準偏差で割ることの意味（グラフ参照） 
正規化されたデータの約68%が-1と1の間に、約95%が-2と2の間に収まる（正規分布の特性）ので見やすい&比較しやすい。

![正規化の解説図](/src/img/norm.png)

### 線形回帰の定義:

- 定義: 独立変数と従属変数の間の線形関係をモデル化する統計的手法
- 目的: データに最もフィットする直線（またはハイパープレーン）を見つけること
- 基本形式:
  - 単回帰: y = ax + b  
  - 多重回帰: y = a1x1 + a2x2 + ... + anxn + b
- 仮定:
  - 線形性: 独立変数と従属変数の関係が線形である
  - 独立性: 観測値が互いに独立している
  - 等分散性: 誤差の分散が一定である
  - 正規性: 誤差が正規分布に従う

### 正規方程式と勾配降下法の比較:

#### アプローチ:  

正規方程式: 一度の計算で厳密解を求める解析的手法  
勾配降下法: 反復的に近似解を求める数値的手法  

#### 計算式:  

正規方程式: θ = (X^T X)^(-1) X^T y  
勾配降下法: θ = θ - α * ∇J(θ) （反復的に適用）  

#### 計算量:  

正規方程式: O(n^3)、nは特徴の数  
勾配降下法: O(kn^2)、kは反復回数、nはデータ点の数  

#### データ規模への適合性:  

正規方程式: 小～中規模のデータセットに適している  
勾配降下法: 大規模なデータセットに適している  

#### メモリ要件:  

正規方程式: すべてのデータを一度にメモリに保持する必要がある  
勾配降下法: データを小バッチで処理できるため、メモリ効率が良い  

#### 収束性:  

正規方程式: 一度の計算で最適解に到達  
勾配降下法: 学習率の設定によっては最適解に収束しない可能性がある  

#### 柔軟性:  

正規方程式: 線形回帰問題に特化  
勾配降下法: 様々な最適化問題に適用可能  
