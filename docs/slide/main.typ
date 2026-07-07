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



// 原稿
// ## p2
// - ニューロンは、脳の情報処理を担う細胞で、外部の電流入力に対し、左図のように特徴的な膜電位応答を行います。
// - そして、ニューロンのこの膜電位応答は、連立の常微分方程式を解くことでシミュレーションできます。このシミュレーションモデルの中には、ゲート変数と呼ばれているいくつかの隠れ変数があることが多いです。例えば、このHHモデルの場合、膜電位の変数に加え、隠れ変数のm,h,mが３つあります。
// ## p3
// - 実際にこれは、脳のシミュレーションを行う際に問題となります。
// - 実際にこの２つの図は実際の脳シミュレーションに用いられた脳モデルですが、とんでも無く複雑です。
// - このように、多くのニューロンを同時にシミュレーションしなければならない場合、メモリが不足します。

// 気をつけること
// 全体像の中で何をやっている中で何をやっているのか、きっちりしたストーリーをキーワードに注目して話すように
// 条件設定は明確にわかりやすく ex.どんなマルチコンパートメントモデルをどうサロゲートモデルにしたのか dt幅はいくつか
// 研究の立ち位置を明確にすること　自分の研究が何を目指しているのか、何を解決しようとしているのかを明確にすること

// 研究報告の事前準備
// スライド作成の際は、流れを先に考え、手書きで軽い構成を練ること
// スライド作成には時間をかけなくてもいいような仕組みを作ること
// 発表の際には研究ノートに軽くスクリプトを書いておくこと　スライドに書かれていないことを補足
// 発表練習は質より量をこなすように
// 月毎の報告では、手法までは簡単な説明でいい　結果をちゃんと話せるようにする

// 表示の際の注意
// 変数は統一し、わかりやすい命名にすること
// 最大コンダクタンスのような最大値を表す定数には上バーがつく
// 斜体が変数や定数 立体が添え字
// スライドの中でI_ionは\sum_x g_x(V,t)(V-E_x) と展開して書くこと

// 用語周りの注意　なるべく略語や専門用語は使わないこと　使う場合は軽い説明をすること　正しい日本語を使うこと
// ハイパラ=>ハイパーパラメーター
// レート関数　もっともんか外にもわかるような説明に
// ' で微分を表すのではなく d/dt で表すようにする
// 近しい=>近い
// 変数 → 状態変数
// ~になってしまいました　だめ　もっと簡潔に　なりました　という


// 改善リスト
// 何度も同じことを言わない　説明は具体的かつ簡潔にまとめて話す　 説明の中に誤魔化を入れないこと 焦らない 
// 今後の最終目的がTraubということを明示
// 結果としてどの程度の再現度と時間・空間計算量の減少（あるいは増加）に繋がったかを説明
// 以下直近の研究報告フィードバック
// 係数どうしの打ち消しが疑われ、数値的不安定さを招きうる ここの意味がわからない
// HHモデルの構造に基づく項をベースに選択　ベースに選択ってどんな感じ?
// 基底関数の選択の仕方が不十分
// 
// 細かなTodo
// 教師電流のseedを変えても問題がないかをちゃんと試してみる
// sin波電流の間隔を縦軸に
// lin_form_expのデバッグ
// ランダムなパルスを組み合わせた電流について、幅もランダムにする