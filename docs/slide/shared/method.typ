== Multi-Compartmentモデルの代理モデルの作り方

#align(center)[
  #block(width: 90%)[
  #image("3comp_surr.png", width: 90%)

]
]


特定のコンパートメントの振る舞いを学習させ、そのモデルを組み込むことで作る

=== コンパートメントの振る舞いの学習のさせ方

1. 学習データを得る
#v(-0.5em)
ニューロンモデル電流を与えてシミュレーションして得られたデータのうち、状態変数のデータを前処理としてPCAで圧縮　　$(V, m, h, n) => (V, g')$

#v(-0.4em)

2. 得られた時系列の学習データ$V,g'$を再現するコンパートメントの代理モデルを SINDy で同定


== SINDyによるダイナミクス同定

時系列データを再現する代理モデルを係数$(a_i, b_i)$を推定することで作成

#grid(
  columns: (1.5fr, auto, 1fr),
  gutter: 1em,
  align: horizon,
  [
    $ C_m (d V) / (d t) = -g_"leak"(V - E_"rest") - I_"ion"(m, h, n) + I_"ext" $
    $ (d x) / (d t) = alpha_x (V)(1 - x) - beta_x (V) x quad (x = m, h, n) $
  ],
  [
    #align(right)[$==>$]
  ],
  [
    $ (d V) / (d t) = sum_i a_i theta_i (V, g', I_"ext") $
    $ (d g') / (d t) = sum_i b_i theta_i (V, g', I_"ext") $
  ],
)

#align(center)[Hodgkin-Huxleyコンパートメントの代理モデル]

#v(1em)

学習前にハイパーパラメーターとしてoptimizerとその設定だけでなく、基底関数$theta$も選択しなければならない
