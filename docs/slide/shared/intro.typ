

== ニューロン数理モデル


#grid(
  columns: (5fr, 4fr),
  gutter: 2em,
  [
    #v(1em)
    #figure(
      image("../pic/previous_research/oridginal_neuron.png", width: 80%),
      caption: [実際のニューロンの膜電位応答(Hodgkin,Huxley 1952)],
    )
    #v(1.5em)
    - ニューロンの膜電位応答は連立の常微分方程式を解くことで知れる
    - 膜電位以外の変数は隠れ変数
  ],
  [
    #figure(
      image("../pic/previous_research/yamazaki.png", width: 80%),
      caption: [Hodgkin-Huxley モデルの膜電位応答(初めての神経回路シミュレーション)],
    )
  ],
)




== 脳シミュレーション時の問題

#grid(
  columns: (3fr, 2fr),
  gutter: 1em,
  image("../pic/previous_research/embodied_1.png", width: 100%),
  image("../pic/previous_research/embodied_2.png", width: 100%),
)

全脳シミュレーションのように多くのニューロンを同時にシミュレーションしなければならない場合、ゲート変数によってメモリが不足

==
ニューロンの空間形状を連結するコンパートメントとしてモデル化しているマルチコンパートメントモデルは、各コンパートメントごとに膜電位と隠れ変数をもつため計算に多くのメモリが必要


#grid(
  columns: (3fr, 2fr),
  gutter: 1em,
  image("../pic/previous_research/compartment.png", width: 95%),
  image("../pic/previous_research/hippocampal_pyramidal.png", width: 40%),
)
$
  C frac(d V_i, d t)=-g_"leak"(V-E_"rest")+I_"ion" + I_"i,i-1" + I_"i,i+1"
$
- 3Compartmentモデル：Hodgkin-huxleyモデルの拡張として作ったMulti-Compartmentモデル

#v(1fr)
目的

3Compartmentモデル
の膜電位応答をより少ないメモリで再現できる代理モデルの作成




// $V_i$($i$番目のコンパートメントの膜電位)の計算には、$I_"i,i-1",I_"i,i+1"$(隣接のコンパートメントから流入出する電流)が必要

== 3コンパートメントモデル
// #image("../pic/3comp.png", width: 100%)

$
  cases(
    C frac(d V_"dend", d t)=-g_"leak"(V-E_"rest") && -I_"pre" && +I_"ext",
    C frac(d V_"soma", d t)=-g_"leak"(V-E_"rest") & + text(#blue,I_"ion-HodgkinHuxley")(V_"soma","gates") & +  I_"pre"&-I_"post" &,
    C frac(d V_"axon", d t)=-g_"leak"(V-E_"rest") &&& +I_"post" &
  )
$
- 電位差に応じて電流がコンパートメント間を流れる($I_"pre",I_"post"$)
- dend,axonは受動的,somaは隣接するコンパートメントから流入した電流によりスパイク発射