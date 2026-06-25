#import "@preview/slydst:0.1.4": *

#set text(font: ("Hiragino Kaku Gothic ProN", "New Computer Modern"))
#show figure.caption: it => it.body
#show link: underline

#show: slides.with(
  title: "研究進捗報告",
  subtitle: "Biophysical ニューロンモデルに対する代理モデル",
  authors: ("Haruki Munechika",),
)

== ニューロン数理モデル

#grid(
  columns: (5fr, 4fr),
  gutter: 2em,
  [
    #v(1em)
    #figure(
      image("./pic/previous_research/oridginal_neuron.png", width: 80%),
      caption: [実際のニューロンの膜電位応答(Hodgkin,Huxley 1952)],
    )
    #v(1.5em)
    - ニューロンの膜電位応答は連立の常微分方程式を解くことで知れる
    - 膜電位以外の変数は隠れ変数
  ],
  [
    #figure(
      image("./pic/previous_research/yamazaki.png", width: 80%),
      caption: [Hodgkin-Huxley モデルの膜電位応答(初めての神経回路シミュレーション)],
    )
  ],
)

== 脳シミュレーション時の問題

#grid(
  columns: (3fr, 2fr),
  gutter: 1em,
  image("./pic/previous_research/embodied_1.png", width: 100%),
  image("./pic/previous_research/embodied_2.png", width: 100%),
)

全脳シミュレーションのように多くのニューロンを同時にシミュレーションしなければならない場合、ゲート変数によってメモリが不足

==
ニューロンの空間形状を連結するコンパートメントとしてモデル化しているマルチコンパートメントモデルは、各コンパートメントごとに膜電位と隠れ変数をもつため計算に多くのメモリが必要


#grid(
  columns: (3fr, 2fr),
  gutter: 1em,
  image("pic/previous_research/compartment.png", width: 95%),
  image("./pic/previous_research/hippocampal_pyramidal.png", width: 40%),
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
#image("3comp.png", width: 100%)

$
  cases(
    C frac(d V_"dend", d t)=-g_"leak"(V-E_"rest") && -I_"pre" && +I_"ext",
    C frac(d V_"soma", d t)=-g_"leak"(V-E_"rest") & + text(#blue,I_"ion-HodgkinHuxley")(V_"soma","gates") & +  I_"pre"&-I_"post" &,
    C frac(d V_"axon", d t)=-g_"leak"(V-E_"rest") &&& +I_"post" &
  )
$
- 電位差に応じて電流がコンパートメント間を流れる($I_"pre",I_"post"$)
- dend,axonは受動的,somaは隣接するコンパートメントから流入した電流によりスパイク発射



== 3Compartment代理モデルの作成方法
#image("3comp.png", width: 100%)

somaコンパートメントはHodgkin-Huxleyモデルと同じように振る舞う

はじめにHodgkin-Huxley代理モデルを作り、これをsomaコンパートメントと置き換える

== Hodgkin-Huxley代理モデルの作成方法

+ Hodgkin-Huxleyモデルに電流を与えてシミュレーション(膜電位V(t)、隠れ変数のデータを得る)
+ 得られた隠れ変数のデータをPCAを用い１次元データ$g_0(t)$に圧縮
+ 時系列データ$V(t),g_0(t)$を再現する代理モデルをSINDyで同定

$
  cases(
    frac(d V, d t) = text(#red, a_11) ( text(#blue,alpha_m) (V) g_0) + text(#red, a_12) ( text(#blue,alpha_m) (V) I_("ext")) - text(#red, a_13) (text(#blue,alpha_m) (g_0) I_("ext")) + dots,
    frac(d g_0, d t) = text(#red, a_21) (text(#blue,alpha_m) (V) g_0) text(#red, a_22) (text(#blue,alpha_n) (V) I_("ext")) +text(#red, a_23)(text(#blue,alpha_m) (V)) + dots
  )
$
- SINDy(非線形ダイナミクスのスパース同定)で、元のダイナミクスを再現するよう上式の#text(red)[係数]推定
- 代理モデル内の#text(blue)[関数]はSINDyにおけるハイパーパラメーターで、Hodgkin-Huxleyモデル中に出現する関数群を選択



