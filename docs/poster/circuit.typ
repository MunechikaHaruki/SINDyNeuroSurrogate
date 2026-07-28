// Traub Multi-Compartment モデル 1 コンパートメントの等価回路
// zap (https://zap.grangelouis.ch) で記号ベースに宣言的に描画。
// 座標を手で追わなくても抵抗/コンデンサ/電池/接地記号が標準の見た目で揃う。
#import "@preview/zap:0.6.0"
#import "@preview/cetz:0.5.2": draw

#let traub_circuit(unit: 0.5cm, label-size: 8pt, stroke-w: 0.7pt) = {
  let lab(body) = text(size: label-size)[#body]

  zap.circuit({
    import zap: *
    import draw: content, set-style
    set-style(zap: (variant: "ieee", stroke: stroke-w))

    let ytop = 7.0 // 膜レール (チャネルが枝分かれする横線)
    let yj = 9.5 // 軸方向抵抗 + I_ext が合流するノード
    let ybot = 0.0 // 接地レール (チャネルの終端が集まる横線)
    let ygnd = -2.5 // 接地記号が合流するノード
    let xj = 2 // ノードの x 位置 (I_ext の真下 = Na チャネルの真上)
    let dx = 4.6 // 枝間隔 (ラベル同士が重ならない十分な幅)

    // --- 隣接コンパートメントとの軸方向結合。1 点のノードに集約する ---
    node("Vim1", (-3.0, yj), fill: false)
    node("Vip1", (xj + 2 * dx - 3.0, yj), fill: false)
    node("J", (xj, yj))
    resistor("gprev", "Vim1", "J", label: lab[$g_(i-1,i)$])
    resistor("gnext", "J", "Vip1", label: lab[$g_(i,i+1)$])
    content("Vim1", lab[$text("Comp")_(i-1)$], anchor: "east", padding: 8pt)
    content("Vip1", lab[$text("Comp")_(i+1)$], anchor: "west", padding: 8pt)

    // --- 外部注入電流: ノードへ直接流し込む ---
    node("Iin", (xj, yj + 0.3), fill: false)
    wire("Iin", "J", i: (content: lab[$I_"ext"$], distance: -100%))

    // --- ノードから膜レールへ 1 本の縦線で接続 ---
    node("M", (xj, ytop))
    wire("J", "M")
    content("M", lab[$V_i$ (intracellular)], anchor: "south-west", padding: 10pt)

    // --- 膜レール ---
    node("Mc", (xj - 2 * dx, ytop))
    node("Mleak", (xj - dx, ytop))
    node("Mna", (xj, ytop))
    node("Mk", (xj + dx, ytop))
    node("Mdots", (xj + 2 * dx, ytop))
    wire("Mc", "Mleak", "Mna", "Mk", "Mdots")

    // --- 接地レール ---
    node("Gc", (xj - 2 * dx, ybot))
    node("Gleak", (xj - dx, ybot))
    node("Gna", (xj, ybot))
    node("Gk", (xj + dx, ybot))
    node("Gdots", (xj + 2 * dx, ybot))
    wire("Gc", "Gleak", "Gna", "Gk", "Gdots")

    // --- 膜容量 + イオン電流の枝 ---
    capacitor("Cm", "Mc", "Gc", label: lab[$C_m$])
    resistor("gL", "Mleak", (rel: (0, -2.6), to: "Mleak"), label: lab[$overline(g)_"L"$])
    battery("EL", (), "Gleak", cells: 1, label: lab[$E_"L"$])
    resistor("gNa", "Mna", (rel: (0, -2.6), to: "Mna"), variable: true, label: lab[$overline(g)_"Na" m^2 h$])
    battery("ENa", (), "Gna", cells: 1, label: lab[$E_"Na"$])
    resistor("gK", "Mk", (rel: (0, -2.6), to: "Mk"), variable: true, label: lab[$overline(g)_"K(DR)" n$])
    battery("EK", (), "Gk", cells: 1, label: lab[$E_"K"$])
    content("Mdots", text(size: label-size + 2pt)[$dots.c$], anchor: "west", padding: 10pt)
    content("Gdots", text(size: label-size + 2pt)[$dots.c$], anchor: "west", padding: 10pt)

    // --- 接地レールから 1 点のノードへ集約し、そこに接地記号 (上の I_ext ノードと対称) ---
    node("G", (xj, ygnd))
    wire("Gna", "G")
    earth("gnd", "G")
    content("G", lab[extracellular], anchor: "west", padding: 10pt)

    // --- V_i = 細胞内 (膜レール) と接地レールの電位差であることを両矢印で明示 ---
    draw.line(
      (xj - 2 * dx - 1.8, ytop),
      (xj - 2 * dx - 1.8, ybot),
      stroke: stroke-w,
      mark: (start: "stealth", end: "stealth", scale: 0.35),
    )
    content((xj - 2 * dx - 1.95, (ytop + ybot) / 2), lab[
    #h(7em)$V_i$ \
      (i th Compartment \ membrane potential)], anchor: "east")
  }, length: unit)
}
