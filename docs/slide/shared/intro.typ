
== ニューロン数理モデル


#grid(
  columns: (6fr, 4fr),
  gutter: 2em,
  align: horizon,
  [
    #v(1em)
    #figure(
      image("./previous_research/oridginal_neuron.png", width: 80%),
      caption: [実際のニューロンの膜電位応答(Hodgkin,Huxley 1952)],
    )
    #v(1.5em)

  ],
  [
    #figure(
      image("./previous_research/yamazaki.png", width: 80%),
      caption: [Hodgkin-Huxley モデルの膜電位応答(初めての神経回路シミュレーション)],
    )
  ],
)
- ニューロンの膜電位応答は連立の常微分方程式を解くことで知れる
- 膜電位以外の変数は隠れ変数



== 脳シミュレーションの問題
#align(center)[
  #block(width: 70%)[
    #grid(
      columns: (7fr, 3fr),
      gutter: 0em,
      align: horizon,
      image("./previous_research/embodied_1.png", width: 100%),
      image("./previous_research/embodied_2.png", width: 100%),
    )
  ]
]

全脳シミュレーションのように多くのニューロンを同時にシミュレーションしなければならない場合、ゲート変数によってメモリが不足


== Multi-Compartmentモデルと研究目的

Multi-Compartmentモデル：空間形状を電気的に連結するコンパートメントとしてモデル化したニューロンモデル（Compartmentごとに状態変数を持つ）

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  align: horizon,
  [
    #image("./previous_research/compartment.png", width: 100%)
    #image("./previous_research/hippocampal_pyramidal.png", width: 85%, height: 60pt, fit: "stretch")
  ],
  [
    #text(size: 14pt)[
      $ I_(i,j) = g_(i,j)(V_j - V_i) $

      $ I_(i("axial")) = sum_(j in "neighbors") g_(i,j)(V_j - V_i) + I_("inj") $
    ]

  ],
)

#v(1em)
目的：Multi-Compartmentモデルの膜電位応答をより少ないメモリで再現できる代理モデルの作成
