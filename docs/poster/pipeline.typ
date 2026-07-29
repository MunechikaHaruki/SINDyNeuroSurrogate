// 学習パイプラインの流れ図 (CeTZ で描画)
// ① MC モデルの 1 comp をシミュレーション → ② ゲートだけ圧縮 → ③ 潜在の式を同定
// → ④ 学習済み decoder/SINDy を等価回路の gate 計算に差し込んでシミュレーション再構成
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
    content((6.0, 5.9), lab[record compartment data], anchor: "west")

    // 膜電位 V (圧縮しないので別枠)
    line(.._spike-pts(6.0, 4.7, 3.4, 0.9, (0.2, 0.45, 0.72)), stroke: 1.6pt + blue)
    content((9.7, 5.1), lab(fill: blue)[$v(t)$], anchor: "west")

    // ゲート変数群
    line(.._smooth-pts(6.0, 3.4, 3.4, 0.8, freq: 1.4, phase: 0.6))
    line(.._smooth-pts(6.0, 2.2, 3.4, 0.8, freq: 1.9, phase: 2.2))
    line(.._smooth-pts(6.0, 1.0, 3.4, 0.8, freq: 1.1, phase: 4.0))
    content((7.7, 0.5), lab[$dots.v$])
    content((9.7, 3.8), lab[$m(t)$], anchor: "west")
    content((9.7, 2.6), lab[$n(t)$], anchor: "west")
    content((9.7, 1.4), lab[$h(t)$], anchor: "west")
    content((12, 2.6), text(size: label-size * 4.0)[\{])
    content((12.3, 2.6), lab[6 gates], anchor: "west")
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
    // line(.._spike-pts(0.1, 6.5, 2.3, 0.9, (0.2, 0.5, 0.78)), stroke: 1.6pt + blue)
    // content((2.6, 6.9), lab(fill: blue)[$V$], anchor: "west")
    // line((3.3, 6.9), (10.6, 6.9), mark: (end: "stealth", scale: 0.5), stroke: 1.6pt + blue)
    // content((6.9, 7.15), lab(fill: blue)[passes through, *not* compressed], anchor: "south")

    // --- ゲート群 ---
    line(.._smooth-pts(0.1, 4.3, 2.3, 0.8, freq: 1.4, phase: 0.6))
    line(.._smooth-pts(0.1, 3.1, 2.3, 0.8, freq: 1.9, phase: 2.2))
    line(.._smooth-pts(0.1, 1.9, 2.3, 0.8, freq: 1.1, phase: 4.0))
    content((1.3, 1.4), lab[$dots.v$])
    content((2.8, 3.3), text(size: label-size * 4.0)[\{])
    content((3.3, 3.8), lab[6 gates], anchor: "west")

    let ae_input_x=5.7
    let ae_output_x=8.7
    let ae_y_middle=3.3
    // --- 圧縮器 (台形 = 次元が絞られる encoder) ---
    line((4.9, 3.3), (5.6, 3.3), mark: (end: "stealth", scale: 0.5))
    line(
      (ae_input_x, ae_y_middle + 1.4),
      (ae_output_x, ae_y_middle + 0.5),
      (ae_output_x, ae_y_middle - 0.5),
      (ae_input_x, ae_y_middle - 1.4),
      close: true,
      stroke: 2.4pt,
    )
    content((7.1, 3.4), lab[AE#v(-2.4em)\ #text(size: label-size * 0.8)[encoder]])
    line((8.8, 3.3), (9.5, 3.3), mark: (end: "stealth", scale: 0.5))

    // --- 潜在変数 ---
    line(.._smooth-pts(9.7, 3.5, 2.0, 0.8, freq: 1.2, phase: 1.0), stroke: 1.6pt + green.darken(25%))
    line(.._smooth-pts(9.7, 2.1, 2.0, 0.8, freq: 0.9, phase: 3.4), stroke: 1.6pt + green.darken(25%))
    content((11.9, 3.9), lab(fill: green.darken(25%))[$z_1 (t)$], anchor: "west")
    content((11.9, 2.5), lab(fill: green.darken(25%))[$z_2 (t)$], anchor: "west")
    content((10.7, 1.6), lab[$dots.v$])
    content((10.7, 1.1), lab[5 latent variables], anchor: "north")
  },
)

// ================= ③ 潜在の支配方程式を同定 =================
// 同定式そのもの (z1, z2 の展開) は canvas でなく main.typ 側で図の下に
// typst 標準 equation として置く (狭い列幅では canvas 内の横長数式が崩れるため)。
#let stage_identify(unit: 1cm, label-size: 22pt) = canvas(
  length: unit,
  {
    import draw: *
    set-style(stroke: 1.4pt, content: (padding: 0.1))
    let lab = (..a) => text(size: label-size, ..a)

    // --- 入力 [V, z] ---
    content((-0.5, 3.4), lab(fill: blue)[$v(t)$], anchor: "west")
    line((0.5, 3.35), (1.5, 2.95), mark: (end: "stealth", scale: 0.5), stroke: 1.6pt + blue)
    content((-0.5, 1.5), lab(fill: green.darken(25%))[$z(t)$], anchor: "west")
    line(
      (0.5, 1.55),
      (1.5, 1.95),
      mark: (end: "stealth", scale: 0.5),
      stroke: 1.6pt + green.darken(25%),
    )

    // --- SINDy: ライブラリ基底 = ゲートのレート関数 (図はここまでに簡略化) ---
    rect((1.55, 1.4), (4.7, 3.5), stroke: 2.4pt)
    content((3.15, 2.4), lab[SINDy\ #text(fill:red,size:label-size*0.9)[rate based basis ]])
    line((4.7, 2.45), (5.5, 2.45), mark: (end: "stealth", scale: 0.5))
    content((5.7, 2.45), lab[$(d z_1) / (d t), dots, (d z_5) / (d t)$], anchor: "west")
  },
)

// ================= ④ decoder/SINDy を等価回路の gate 計算に差し込んでシミュレーション =================
// hybrid_kernel.py の 1 ステップ: decode(z) → gates → 元の dV/dt (等価回路, 不変) と
// 並行して SINDy が dz/dt を返す → 積分して次ステップへ。等価回路の"構造"自体は
// 一切変えず、ゲート計算部分だけを学習済み decoder+SINDy に差替えていることを示す。
// CeTZ の手動座標は狭い列幅で崩れやすいので、typst 標準の box + 矢印テキストで組む
// (自動サイズ調整に任せる)。
#let _flow-box(body, color: black) = box(
  stroke: 1.4pt + color,
  inset: 5pt,
  radius: 3pt,
  baseline: 50%,
  body,
)
#let _arrow(size: 18pt) = text(size: size)[#sym.arrow.r]
