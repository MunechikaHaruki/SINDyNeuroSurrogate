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

// 数値文字列を小数第2位まで丸める（非数値はそのまま返す）
#let fmt(v) = {
  // 前後の空白を除去
  let trimmed = v.trim()
  // 整数・小数・科学記法を判定する正規表現
  if trimmed.match(regex("^-?[0-9]+(\.[0-9]+)?(e[+-]?[0-9]+)?$")) != none {
    // 数値なら小数第2位に丸めて文字列に変換
    str(calc.round(float(trimmed), digits: 2))
  } else {
    // 非数値（ヘッダー等）はそのまま返す
    trimmed
  }
}

// CSV読み込み＋数値フォーマット＋booktabs描画
// 指定した行インデックス（0始まり、ヘッダー除く）だけを抜き出して描画
#let booktabs(path, rows: none, offset: -2) = {
  let data = csv(path)
  let header = data.at(0)
  // rows指定があればその行だけ、なければ全行
  let all_rows = data.slice(1)
  let selected = if rows == none {
    all_rows
  } else {
    rows.map(i => all_rows.at(i + offset))
  }
  table(
    columns: header.len(),
    stroke: none,
    inset: (x: 6pt, y: 5pt),
    align: (col, _) => if col == 0 { left } else { right },
    table.hline(stroke: 1.2pt),
    ..header.map(h => text(weight: "bold", size: 0.78em, h)),
    table.hline(stroke: 0.6pt),
    ..selected.flatten().map(v => text(size: 0.78em, fmt(v))),
    table.hline(stroke: 1.2pt),
  )
}


== 単一スパイクの比較
オリジナルのHHモデル、baseモデルに10ms秒の振幅10mAの定常電流を与えた
#grid(
  columns: (1.3fr, 1fr),
  gutter: 0.3em,
  image("result/single_waveform.png", width: 90%),
  {
    booktabs("result/single_metrics.csv",rows:(3,))
    v(1.2em)
    booktabs("result/single_scalar_metrics.csv")
  }
)
== 振幅を変えた時の振る舞い
a
== 定常電流以外の入力

== 5compモデル
a

== まとめ
まとめ


#include "shared/appendix.typ"
