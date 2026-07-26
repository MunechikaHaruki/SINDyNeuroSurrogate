// 学習パイプラインの流れ図 (CeTZ で描画)
// ① MC モデルの 1 comp をシミュレーション → ② ゲートだけ圧縮 → ③ 潜在の式を同定
#import "@preview/cetz:0.4.2": canvas, draw

// ---- 波形ヘルパ (すべて (x,y) 点列を返す) ----

// スパイク列 (膜電位っぽい波形)
#let _spike-pts(x0, y0, w, h, centers, n: 80) = range(0, n + 1).map(i => {
  let t = i / n
  let v = 0.0
  for c in centers {
    v += calc.exp(-calc.pow((t - c) * 26, 2))
  }
  (x0 + t * w, y0 + h * calc.min(v, 1.0))
})

// なめらかなゲート波形
#let _smooth-pts(x0, y0, w, h, freq: 1.5, phase: 0.0, n: 60) = range(0, n + 1).map(i => {
  let t = i / n
  (x0 + t * w, y0 + h * 0.5 * (1 + calc.sin(2 * calc.pi * freq * t + phase)))
})

// パルス列 (注入電流)
#let _pulse-pts(x0, y0, w, h, levels) = {
  let pts = ()
  for (i, lv) in levels.enumerate() {
    pts.push((x0 + w * i / levels.len(), y0 + h * lv))
    pts.push((x0 + w * (i + 1) / levels.len(), y0 + h * lv))
  }
  pts
}

// ================= ① シミュレーションと時系列の収集 =================
#let stage_simulate(unit: 1cm, label-size: 22pt) = canvas(
  length: unit,
  {
    import draw: *
    set-style(stroke: 1.4pt, content: (padding: 0.1))
    let lab = (..a) => text(size: label-size, ..a)

    // --- 注入電流 I_ext ---
    content((0.1, 9.0), lab[$I_"ext"$], anchor: "west")
    line(
      .._pulse-pts(1.9, 8.5, 7.0, 1.0, (0, 0.7, 0.2, 1.0, 0.1, 0.55, 0.9, 0.15, 0.6, 0)),
      stroke: 1.4pt + orange,
    )
    line((5.4, 8.3), (5.4, 7.6), mark: (end: "stealth", scale: 0.5))

    // --- コンパートメント鎖 (刺激は soma へ、記録は全 comp から) ---
    let cw = 1.45
    for i in range(7) {
      let x = 0.4 + i * cw
      if i == 3 {
        rect((x, 6.4), (x + cw * 0.82, 7.5), fill: rgb("#cfe3ff"), stroke: 2.4pt)
      } else {
        rect((x, 6.6), (x + cw * 0.82, 7.3))
      }
      if i < 6 { line((x + cw * 0.82, 6.95), (x + cw, 6.95)) }
    }
    line((4.6, 6.4), (3.6, 5.7))
    content((3.5, 5.65), lab[soma], anchor: "east")

    // --- 全 comp から時系列を取り出す (学習軌道 = 19 comp 全部) ---
    line((5.3, 6.4), (5.3, 5.9), (5.8, 5.5), mark: (end: "stealth", scale: 0.5))
    content((6.0, 5.7), lab[record *every* compartment], anchor: "west")

    // 膜電位 V (圧縮しないので別枠)
    line(.._spike-pts(6.0, 4.7, 3.4, 0.9, (0.2, 0.45, 0.72)), stroke: 1.6pt + blue)
    content((9.7, 5.1), lab(fill: blue)[$V$], anchor: "west")

    // ゲート変数群
    line(.._smooth-pts(6.0, 3.4, 3.4, 0.8, freq: 1.4, phase: 0.6))
    line(.._smooth-pts(6.0, 2.2, 3.4, 0.8, freq: 1.9, phase: 2.2))
    line(.._smooth-pts(6.0, 1.0, 3.4, 0.8, freq: 1.1, phase: 4.0))
    content((7.7, 0.5), lab[$dots.v$])
    content((9.7, 3.8), lab[$M$], anchor: "west")
    content((9.7, 2.6), lab[$N$], anchor: "west")
    content((9.7, 1.4), lab[$C$], anchor: "west")
    content((10.6, 2.6), text(size: label-size * 4.0)[\{])
    content((11.1, 2.6), lab[10 gates], anchor: "west")
  },
)

// ================= ② ゲートだけを潜在へ圧縮 =================
#let stage_compress(unit: 1cm, label-size: 22pt) = canvas(
  length: unit,
  {
    import draw: *
    set-style(stroke: 1.4pt, content: (padding: 0.1))
    let lab = (..a) => text(size: label-size, ..a)

    // --- V は圧縮せず素通し ---
    line(.._spike-pts(0.1, 6.5, 2.3, 0.9, (0.2, 0.5, 0.78)), stroke: 1.6pt + blue)
    content((2.6, 6.9), lab(fill: blue)[$V$], anchor: "west")
    line((3.3, 6.9), (10.6, 6.9), mark: (end: "stealth", scale: 0.5), stroke: 1.6pt + blue)
    content((6.9, 7.15), lab(fill: blue)[passes through, *not* compressed], anchor: "south")

    // --- ゲート群 ---
    line(.._smooth-pts(0.1, 4.3, 2.3, 0.8, freq: 1.4, phase: 0.6))
    line(.._smooth-pts(0.1, 3.1, 2.3, 0.8, freq: 1.9, phase: 2.2))
    line(.._smooth-pts(0.1, 1.9, 2.3, 0.8, freq: 1.1, phase: 4.0))
    content((1.3, 1.4), lab[$dots.v$])
    content((2.8, 3.3), text(size: label-size * 4.0)[\{])
    content((3.2, 3.3), lab[6 gates], anchor: "west")

    // --- Ca²⁺ サブ系 (S, R, Q, XI) も圧縮せず physics で解く ---
    content((0.1, 0.5), lab[$S, R, Q, xi$], anchor: "west")
    line((2.4, 0.5), (9.3, 0.5), mark: (end: "stealth", scale: 0.5), stroke: 1.4pt + gray)
    content((5.8, 0.7), lab(fill: gray.darken(35%))[$"Ca"^(2+)$ subsystem: also physics], anchor: "south")

    // --- 圧縮器 ---
    line((4.9, 3.3), (5.6, 3.3), mark: (end: "stealth", scale: 0.5))
    rect((5.7, 2.4), (8.7, 4.2), stroke: 2.4pt)
    content((7.2, 3.3), lab[PCA / AE])
    line((8.8, 3.3), (9.5, 3.3), mark: (end: "stealth", scale: 0.5))

    // --- 潜在変数 ---
    line(.._smooth-pts(9.7, 3.5, 2.0, 0.8, freq: 1.2, phase: 1.0), stroke: 1.6pt + green.darken(25%))
    line(.._smooth-pts(9.7, 2.1, 2.0, 0.8, freq: 0.9, phase: 3.4), stroke: 1.6pt + green.darken(25%))
    content((11.9, 3.9), lab(fill: green.darken(25%))[$z_1$], anchor: "west")
    content((11.9, 2.5), lab(fill: green.darken(25%))[$z_2$], anchor: "west")
    content((10.7, 1.6), lab[$dots.v$])
    content((10.7, 1.1), lab[$bold(z) in RR^n$], anchor: "north")
  },
)

// ================= ③ 潜在の支配方程式を同定 =================
#let stage_identify(unit: 1cm, label-size: 22pt) = canvas(
  length: unit,
  {
    import draw: *
    set-style(stroke: 1.4pt, content: (padding: 0.1))
    let lab = (..a) => text(size: label-size, ..a)

    // --- 入力 [V, z] ---
    content((0.1, 4.9), lab(fill: blue)[$V$], anchor: "west")
    line((0.7, 4.85), (1.8, 4.45), mark: (end: "stealth", scale: 0.5), stroke: 1.6pt + blue)
    content((0.1, 3.0), lab(fill: green.darken(25%))[$bold(z)$], anchor: "west")
    line(
      (0.7, 3.05),
      (1.8, 3.45),
      mark: (end: "stealth", scale: 0.5),
      stroke: 1.6pt + green.darken(25%),
    )

    // --- SINDy ---
    rect((1.9, 2.9), (4.6, 5.0), stroke: 2.4pt)
    content((3.25, 3.95), lab[SINDy])
    line((4.7, 3.95), (5.5, 3.95), mark: (end: "stealth", scale: 0.5))

    // --- 同定される潜在方程式 (基底の 1 項を丸で囲むため content を分割) ---
    content((5.7, 4.6), lab[$frac(d z_1, d t) = xi_11$], anchor: "west", name: "e1")
    content("e1.east", lab[$ alpha_M (V) $], anchor: "west", name: "basis")
    content(
      "basis.east",
      lab[#h(0.35em)$+ xi_12 beta_M (V) z_1 + dots.c$],
      anchor: "west",
    )
    content((5.7, 3.0), lab[$frac(d z_2, d t) = xi_21 alpha_N (V) + dots.c$], anchor: "west")
    content((6.6, 2.1), lab[$dots.v$])

    // --- ライブラリ基底 = ゲートのレート関数 ---
    circle("basis", radius: (0.95, 0.45), stroke: 1.8pt + red)
    line((8.2, 5.2), (8.9, 6.2), mark: (end: "stealth", scale: 0.5), stroke: 1.6pt + red)
    content(
      (9.0, 6.3),
      lab(fill: red)[library basis =\ the gates' own\ *rate functions*],
      anchor: "south-west",
    )
  },
)
