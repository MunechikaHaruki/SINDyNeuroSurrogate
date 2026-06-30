#import "@preview/slydst:0.1.4": *
#import "shared/style.typ": booktabs
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

#let image_path = "/docs/slide/result/"

== Compartment学習の条件

#columns(2)[

  *教師電流:*

  −5〜20 mA の、20 ms 幅のランダムな振幅を持つパルス電流

  #figure(image("shared/current_teaching.png"))

  #colbreak()

  #v(4em)
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
  #image(image_path + "model.png")
]

== 単一スパイクの比較
オリジナルのHHモデル、baseモデルに10ms秒の振幅10mAの定常電流を与えた
#grid(
  columns: (1.3fr, 1.5fr),
  gutter: -1.5em,
  image(image_path + "single_waveform.png", width: 90%),
  {
    booktabs(image_path + "single_metrics.csv",rows:(3,7,8,11),size: 0.67em)
    v(-0.3em)
    booktabs(image_path + "single_scalar_metrics.csv")
  }
)
== 振幅を変えた時の振る舞い
100ms間の定常電流を振幅を変えながら加えた時のスパイク数の推移
#grid(columns:(1fr,1fr),gutter:0.5em,
  figure(image(image_path+"amptitude_waveform.png"),
  caption: "baseモデルとオリジナルモデルに10mAの定常電流付加"),
  
  image(image_path+"amptitude_sweep.png")
  )

== 閾値付近での振る舞い
閾値

#grid(columns:(1fr,1fr),gutter:0.5em,
figure(image(image_path+"ramp_waveform.png"),)
  )


== 定常電流以外の入力
sin波電流を与える
#grid(columns:(1fr,1fr),gutter:0.5em,
  figure(image(image_path+"sinousoidal_waveform.png"),
  caption: "50Hz"),
  
  image(image_path+"sinousoidal_sweep.png")
  )

==　5Compモデル
h0,h1,h2をbaseモデルに置換
// 上部: タイトル行とネットワーク図を横並びに

#place(top+right,dx: -8em,dy: -3em)[

  #image("result/5comp_neurograph.png", height: 2.2cm)
]
#v(1em)
// 下部: 左に4パネルの時系列図、右にスパイクカウント図
#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  [
    #image("result/5comp_waveform.png", width: 90%)
  ],
  [
    #image("result/5comp_sweep.png", width: 90%)
    #v(4pt)
    #align(center)[
      #text(size: 12pt)[100msの定常電流を与えた時のスパイク数]
    ]
  ]
)


== まとめ
まとめ


#include "shared/appendix.typ"
