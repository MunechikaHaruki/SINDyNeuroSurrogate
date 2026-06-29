#import "@preview/slydst:0.1.4": *

#set text(font: ("Hiragino Kaku Gothic ProN", "New Computer Modern"))
#show figure.caption: it => it.body
#show link: underline
#show figure.caption: set text(size: 9pt)
#set text(size: 12pt)

#show: slides.with(
  title: "研究進捗報告",
  subtitle: "Biophysical ニューロンモデルに対する代理モデル",
  authors: ("Haruki Munechika",),
  ratio: 16 / 9,
)

#include "shared/intro.typ"

#include "shared/method.typ"


== Compartment学習の条件

#columns(2)[

  *教師電流:*

  −5〜20 mA の、20 ms 幅のランダムな振幅を持つパルス電流

  #figure(image("shared/current_teaching.png"))

  #colbreak()

  - *optimizer*: STLSQ なオプティマイザ
  - *solver*: 陽的 Euler 法　$Delta t = 0.01$ [ms]
]
  *基底関数*

  HH モデルを構成するレート関数と、HH モデルの構造に基づく項を
  ベースに選択

  *ex.* $quad alpha_m (V) g quad$：$d v(m, t) = alpha_m (1 - m) - beta_m dot m$ ,
  $quad V,; 1 quad$：定数項


== 基底関数の種類

基底関数は3セットほど用意して、各場合について学習させた

#v(0.8em)

#block(inset: (left: 1em))[
  *base*：$alpha_m (V) g$ の他にも、$alpha_m (V) I_"ext"$ のような不自然な項を含む

  #v(0.6em)

  *informed*：$alpha_m (V) I_"ext"$ のような不自然な項を排除

  #v(0.6em)

  *relaxation*：$d v(g_0, t) = alpha_k (V) - (alpha_k (V) + beta_k (V)) dot g_0$
  という緩和形式が基底に陽に含まれている
]

#v(1em)

#align(center)[
  性能は *base* が最良　　基本は base モデルでの結果を示す
]

== 結果 得られたモデル
#v(5em)
#align(center)[
  #image("result/model.png")
]

== 単一スパイクとの比較
Write Result Here

== 振幅を変えた時の振る舞い
a
== 定常電流以外の入力

== 5compモデル
a

== まとめ
まとめ


#include "shared/appendix.typ"
