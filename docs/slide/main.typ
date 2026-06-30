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

  HH モデルを構成するレート関数と、HH モデルの構造に基づく項をベースに選択

  *ex.*

  ゲート変数が従う式$(d m) / (d t)  = alpha_m (V) (1 - m) - beta_m (V) m$から$alpha_m (V) g$を選択　
  $V,g,I_"ext",1 $


== 基底関数の種類
基底関数は3セットほど用意して、各場合について学習させた

#v(0.8em)

#block(inset: (left: 1em))[
  *base*：$alpha_m (V) g$ の他にも、$alpha_m (V) I_"ext"$ のような不自然な項を含む

  #v(0.6em)

  *informed*：$alpha_m (V) I_"ext"$ のような不自然な項を排除

  #v(0.6em)

  *relaxation*：$(d m)/(d t) = alpha_m (V) - (alpha_m (V) + beta_m (V)) dot m$
  のような緩和形式から基底を選択
  #v(-0.4em)
  ( $(alpha_m (V)+beta_m (V) )dot g$　のような項を基底とする informedより基底の数が少なくなり、束縛される)
]

#v(1em)

#align(center)[
  性能は *base* が最良　　基本は base モデルでの結果を示す
]

== 結果 得られたモデル

baseモデル(不自然な項も含むモデル)

#align(center)[
  #image(image_path + "model.png")
]

  - HH のレート関数に基づく項が支配的に選択
  - 一部の項の係数が $plus.minus 10^3$ と桁違いに大きい(Symlog scaleのヒートマップ)
  #sym.arrow.r 係数どうしの打ち消しが疑われ、数値的不安定さを招きうる //ここの意味がわからない

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

スパイク1本の形状的な特徴(振幅・幅)は代理モデルである程度再現できている
#v(-0.9em)
rmseに見れるように、位相のずれが生じている

== 振幅を変えた時の振る舞い
100ms間の定常電流を振幅を変えながら加えた時のスパイク数の推移
#grid(columns:(1fr,1fr),gutter:0.5em,
  figure(image(image_path+"amptitude_waveform.png"),
  caption: "baseモデルとオリジナルモデルに10mAの定常電流付加"),
  
  image(image_path+"amptitude_sweep.png")
  )

== 閾値

0から20mAまでのramp電流を100msの区間で与えた

#grid(columns:(1fr,1fr),gutter:0.5em,
figure(image(image_path+"ramp_waveform.png",width: 90%),)
  )

baseモデルは45ms付近、オリジナルモデルは70ms付近で発火 閾値には5mA程度の差が生じている


== 定常電流以外の入力
sin波電流を与える
#grid(columns:(1fr,1fr),gutter:0.5em,
  figure(image(image_path+"sinousoidal_waveform.png"),
  caption: "50Hz"),
  
  image(image_path+"sinousoidal_sweep.png")
  )

==　5Compモデル
h0,h1,h2をbaseモデルに置換

#place(top+right,dx: -8em,dy: -3em)[

  #image("result/5comp_neurograph.png", height: 2.2cm)
]
#v(1em)
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
  *今回の進捗: 代理モデルの評価指標を拡充*

  *スパイク単体の評価*
  - latency・AP_amplitude・AP_begin_voltage・AP_duration_half_width で
    オリジナル HH モデルと定量比較
  - 単一スパイク波形の誤差を RMSE・MAE で評価 (RMSE 17.27, MAE 7.2)
  - AP振幅は約 3 mV の過小評価、その他の指標は概ね一致

  *入力条件を変えた応答の評価*
  - 定常電流の振幅を掃引し、スパイク数 (発火頻度) の推移を比較
  - sin 波電流に対する応答を確認

  *見えてきた限界と今後の方針*
  - 純粋に SINDy で同定する現状の手法では、波形の位相ズレや振幅過小、
    入力条件を変えた際の再現性に限界が見えてきた
  - 今後は元の HH 方程式の構造をより陽に取り込んだ形へ発展させ、
    精度と外挿性能の向上を目指す


#include "shared/appendix.typ"


// Neuroscienceで読むべき本
// Methods of Neuronal Modeling
// Theoretical Neuroscience
// Dynamical System in Neuroscience
// 
// 
// 原稿
// ## p2
// - ニューロンは、脳の情報処理を担う細胞で、外部の電流入力に対し、左図のように特徴的な膜電位応答を行います。
// - そして、ニューロンのこの膜電位応答は、連立の常微分方程式を解くことでシミュレーションできます。このシミュレーションモデルの中には、ゲート変数と呼ばれているいくつかの隠れ変数があることが多いです。例えば、このHHモデルの場合、膜電位の変数に加え、隠れ変数のm,h,mが３つあります。
// ## p3
// - 実際にこれは、脳のシミュレーションを行う際に問題となります。
// - 実際にこの２つの図は実際の脳シミュレーションに用いられた脳モデルですが、とんでも無く複雑です。
// - このように、多くのニューロンを同時にシミュレーションしなければならない場合、メモリが不足します。


// 発表に対するコメント
// - 日本語が変
// - ストーリーを明確に　キーワードに注目して話すように
// - 'は微分を表す
// - 条件設定がわかりにくい
//   - 変数がわかりにくい
//   - 手法の概要を話す
//   - 基底関数についての説明が不十分
//   - param の設定が前と同じことを示す

// - 相関の条件は正当化できるか(定常部分を覗いてみてまた試す)

// 研究の立ち位置がわかりにくい
// 変数 → 状態変数
// 実装に成功？]

// progress.md は todo がわかるようにする

// 省略形は使わない
// 具体的に
// ただし長くしない

// - 流れを先に考える　まずはスライドを手書きして
// - 最終目標はTraubということを書く
// - dend部をdendriteに
// - ハイパラをハイパーパラメーターに
// - 何回も説明しない
// - 誤魔化さない
// - スライド準備に時間をかける(1week前から)

// 全体像の中で何をやっているか
// ストーリーをちゃんと作って説明する

// - マルチコンパートメントモデルのSurrogateモデルについて，（簡単な）イントロと手法，結果の説明をしてください。特に，研究の目的，どんなマルチコンパートメントモデルを何を使ってSurrogateモデルにしたのか，結果としてどの程度の再現度と時間・空間計算量の減少（あるいは増加）に繋がったかを説明してください

// 
// 6/30 研究報告 フィードバック
// 基底関数の選択の仕方が不十分 
// レート関数 用語が専門的すぎる 軽い説明が必要
// ベース 単語がダメ　もっと単純に
// ~になってしまいました　だめ　もっと簡潔に　なりました　という
// spike_countは縦軸が0始まりでなければならない
// 同じことを何度も言わない　簡潔に情報をまとめて話す
// 