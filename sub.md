# ft_linear_regression
機械学習入門

**概要**: このプロジェクトでは、あなたの最初の機械学習アルゴリズムを実装します。
バージョン: 4

## 目次
I. 前書き
II. はじめに
III. 目的
IV. 一般的な指示
V. 必須部分
VI. ボーナス部分
VII. 提出とピア評価

## I. 前書き
私が考える機械学習の最良の定義：
「コンピュータプログラムは、タスクTのクラスと性能指標Pに関して、経験Eから学習すると言える。もし、TにおけるタスクのプログラムのパフォーマンスがPによって測定され、経験Eとともに向上する場合。」
- トム・M・ミッチェル

## II. はじめに
機械学習は、複雑で数学者だけのものと思われがちな成長中のコンピュータサイエンスの分野です。ニューラルネットワークやk-meansクラスタリングについて聞いたことがあるかもしれませんが、それらがどのように機能するのか、またはそのようなアルゴリズムをどのようにコーディングするのかわからないかもしれません...しかし心配しないでください。実際、私たちはシンプルで基本的な機械学習アルゴリズムから始めます。

## III. 目的
このプロジェクトの目的は、機械学習の基本的な概念を紹介することです。
このプロジェクトでは、勾配降下法アルゴリズムで訓練された線形関数を使用して、車の価格を予測するプログラムを作成する必要があります。
プロジェクトでは具体的な例に取り組みますが、完了すれば他のデータセットでもアルゴリズムを使用できるようになります。

## IV. 一般的な指示
このプロジェクトでは、好きな言語を使用することができます。
また、すべての作業をあなたの代わりに行わない限り、どのようなライブラリでも自由に使用できます。例えば、Pythonのnumpy.polyfitの使用はカンニングとみなされます。
データを簡単に視覚化できる言語を使用する必要があります：デバッグに非常に役立ちます。

## V. 必須部分
単一の特徴（この場合、車の走行距離）を用いた単純な線形回帰を実装します。
そのために、2つのプログラムを作成する必要があります：

1. 最初のプログラムは、与えられた走行距離に対する車の価格を予測するために使用されます。
   プログラムを起動すると、走行距離の入力を求め、その走行距離に対する推定価格を返します。プログラムは以下の仮説を使用して価格を予測します：

   estimatePrice(mileage) = θ0 + (θ1 * mileage)

   トレーニングプログラムの実行前は、theta0とtheta1は0に設定されます。

2. 2番目のプログラムは、モデルをトレーニングするために使用されます。データセットファイルを読み込み、データに対して線形回帰を実行します。
   線形回帰が完了したら、最初のプログラムで使用するためにtheta0とtheta1の変数を保存します。

以下の公式を使用します：

tmpθ0 = learningRate * (1/m) * Σ(estimatePrice(mileage[i]) - price[i])
tmpθ1 = learningRate * (1/m) * Σ(estimatePrice(mileage[i]) - price[i]) * mileage[i]

mが何を表すか推測してみてください :)

注意：estimatePriceは最初のプログラムと同じですが、ここでは一時的に最後に計算されたtheta0とtheta1を使用します。
また、theta0とtheta1を同時に更新することを忘れないでください。

## VI. ボーナス部分
非常に有用な可能性のあるボーナスがいくつかあります：

- データをグラフにプロットして、その分布を確認する。
- 線形回帰の結果である直線を同じグラフにプロットして、あなたの努力の結果を確認する！
- アルゴリズムの精度を計算するプログラム。

ボーナス部分は、必須部分が完璧な場合にのみ評価されます。完璧とは、必須部分が完全に行われ、誤動作なく機能することを意味します。必須要件をすべて満たしていない場合、ボーナス部分は一切評価されません。

## VII. 提出とピア評価
通常通り、Gitリポジトリに課題を提出してください。防衛の際には、リポジトリ内の作業のみが評価されます。フォルダとファイルの名前が正しいことを確認するために、二重チェックを躊躇しないでください。

以下は、ピア評価者がチェックする点です：

- あなたの代わりに作業を行うライブラリがないこと
- 指定された仮説の使用
- 指定されたトレーニング関数の使用