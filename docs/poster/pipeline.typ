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

    // --- 圧縮器 ---
    line((4.9, 3.3), (5.6, 3.3), mark: (end: "stealth", scale: 0.5))
    rect((5.7, 2.4), (8.7, 4.2), stroke: 2.4pt)
    content((7.2, 3.3), lab[ AE])
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
      lab(fill: red)[basal function =\ the gates' own\ *rate functions*],
      anchor: "south-west",
    )
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

// ④ 1本の処理の流れとして描く: z→decode(②の鏡写し)→gates→等価回路→V̇、
// そのすぐ下 (等価回路と同じ x 位置) に SINDy(③で学習した ξ) を置き z,V→ż を出す。
// V̇ と ż は右端の integrate box に合流し、次ステップの z,V へループする。
#let stage_simulate_loop(unit: 1cm, label-size: 22pt, body-size: 20pt) = canvas(
  length: unit,
  {
    import draw: *
    set-style(stroke: 1.4pt, content: (padding: 0.1))
    let lab = (..a) => text(size: label-size, ..a)
    // gates 出力以降を右へずらして "6 gates" ラベルと equiv.circuit box の衝突を解消
    let dx = 1.4

    // --- V は decode を経由せずそのまま等価回路へ (②の「passes through」と対称) ---
    line((0.1, 4.6), (11.8 + dx, 4.6), mark: (end: "stealth", scale: 0.5), stroke: 1.6pt + blue)
    content((0.1, 4.75), lab(fill: blue)[$V$ unchanged], anchor: "west")

    // --- 潜在変数 z (②の出力と同じ波形・同じ位置) ---
    line(.._smooth-pts(0.1, 3.5, 2.0, 0.8, freq: 1.2, phase: 1.0), stroke: 1.6pt + green.darken(25%))
    line(.._smooth-pts(0.1, 2.1, 2.0, 0.8, freq: 0.9, phase: 3.4), stroke: 1.6pt + green.darken(25%))
    content((2.3, 3.9), lab(fill: green.darken(25%))[$z_1$], anchor: "west")
    content((1.1, 1.6), lab[$dots.v$])
    content((1.1, 1.1), lab[$bold(z) in RR^n$], anchor: "north")

    // --- decoder (② AE box と同じ大きさ, 逆向き矢印) ---
    line((2.5, 3.3), (3.2, 3.3), mark: (end: "stealth", scale: 0.5))
    rect((3.3, 2.4), (6.3, 4.2), stroke: 2.4pt)
    content((4.8, 3.3), lab[decode #linebreak() #text(size: 0.7em)[AE, reverse of ②]])
    line((6.4, 3.3), (7.1, 3.3), mark: (end: "stealth", scale: 0.5))

    // --- gates (②の入力ゲート群と同じ波形・同じ本数) ---
    line(.._smooth-pts(7.3, 4.3, 2.3, 0.8, freq: 1.4, phase: 0.6))
    line(.._smooth-pts(7.3, 3.1, 2.3, 0.8, freq: 1.9, phase: 2.2))
    line(.._smooth-pts(7.3, 1.9, 2.3, 0.8, freq: 1.1, phase: 4.0))
    content((8.5, 1.4), lab[$dots.v$])
    content((9.9, 3.3), text(size: label-size * 4.0)[\}])
    content((10.3, 3.3), lab[6 gates], anchor: "west")

    // --- 等価回路 (不変, ③の red と同じ色で「触っていない」ことを示す) ---
    line((11.1 + dx, 3.3), (11.8 + dx, 3.3), mark: (end: "stealth", scale: 0.5))
    rect((11.9 + dx, 2.3), (14.9 + dx, 4.3), stroke: 2.4pt + rgb("#8a2020"))
    content((13.4 + dx, 3.7), lab(fill: rgb("#8a2020"))[equiv.#linebreak()circuit])
    content((13.4 + dx, 2.7), lab(fill: rgb("#8a2020"), size: label-size * 0.65)[$dot(V) = f(dot)$,#linebreak()*unchanged*])
    line((15.0 + dx, 3.3), (15.7 + dx, 3.3), mark: (end: "stealth", scale: 0.5))
    content((15.8 + dx, 3.3), lab[$dot(V)$], anchor: "west")

    // --- SINDy: 等価回路の真下 (同じ x 位置), 同じ z, V から ż を出す ---
    line((0.7, 3.0), (0.7, 1.1), (11.1 + dx, 1.1), mark: (end: "stealth", scale: 0.5), stroke: 1.6pt + green.darken(25%))
    line((0.1, 4.6), (0.1, 0.3), (11.1 + dx, 0.3), mark: (end: "stealth", scale: 0.5), stroke: 1.6pt + blue)
    rect((11.9 + dx, -0.1), (14.9 + dx, 1.5), stroke: 2.4pt)
    content((13.4 + dx, 1.0), lab[SINDy $xi$])
    content((13.4 + dx, 0.4), lab(size: label-size * 0.65)[learned in ③])
    line((15.0 + dx, 0.7), (15.7 + dx, 0.7), mark: (end: "stealth", scale: 0.5))
    content((15.8 + dx, 0.7), lab[$dot(bold(z))$], anchor: "west")

    // --- V̇, ż → integrate → 次ステップの z, V へ (1 本の流れとして閉じる) ---
    line((16.6 + dx, 3.3), (17.4 + dx, 1.9), mark: (end: "stealth", scale: 0.5), stroke: 1.6pt + rgb("#8a2020"))
    line((16.9 + dx, 0.7), (17.4 + dx, 1.6), mark: (end: "stealth", scale: 0.5), stroke: 1.6pt + green.darken(25%))
    rect((17.5 + dx, 1.15), (20.5 + dx, 2.65), stroke: 2.4pt)
    content((19.0 + dx, 1.9), lab[integrate #linebreak() #text(size: 0.7em)[(Euler)]])
    line((20.6 + dx, 1.9), (21.3 + dx, 1.9), mark: (end: "stealth", scale: 0.5))
    content((21.4 + dx, 1.9), lab[$bold(z)(t+Delta t), V(t+Delta t)$], anchor: "west")
  },
)
