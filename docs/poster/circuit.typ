// Traub Multi-Compartment モデル 1 コンパートメントの等価回路 (CeTZ で描画)
#import "@preview/cetz:0.4.2": canvas, draw

#let traub_circuit(unit: 0.5cm, label-size: 8pt, stroke-w: 0.7pt) = canvas(
  length: unit,
  {
    import draw: *
    set-style(stroke: stroke-w, content: (padding: 0.06))

    let ytop = 6.0 // 細胞内 (膜電位 V_i) のレール
    let ybot = 0.0 // 細胞外 (基準電位)

    // 可変コンダクタンス (斜め矢印つき抵抗)
    let vres(x, y, var: true) = {
      rect((x - 0.32, y - 0.62), (x + 0.32, y + 0.62), fill: white)
      if var {
        line(
          (x - 0.62, y - 0.85),
          (x + 0.62, y + 0.85),
          mark: (end: "stealth", scale: 0.35),
        )
      }
    }
    // 電池 (反転電位)
    let batt(x, y) = {
      line((x - 0.5, y + 0.16), (x + 0.5, y + 0.16))
      line((x - 0.24, y - 0.16), (x + 0.24, y - 0.16))
    }
    // イオン電流の枝: 可変コンダクタンス + 反転電位
    let ion(x, gl, el, var: true) = {
      line((x, ytop), (x, 4.82))
      vres(x, 4.2, var: var)
      line((x, 3.58), (x, 2.16))
      batt(x, 1.9)
      line((x, 1.74), (x, ybot))
      content((x + 0.75, 4.2), text(size: label-size)[#gl], anchor: "west")
      content((x + 0.75, 1.9), text(size: label-size)[#el], anchor: "west")
    }

    let x-c = 0.0 // 膜容量
    let x-leak = 3.4 // 漏れ
    let x-na = 7.4 // Na
    let x-k = 11.8 // K(DR)
    let x-dots = 16.4 // 残りのイオンチャネル (Ca, K(A), K(AHP), K(C))

    // --- 上下のレール ---
    line((x-c - 1.0, ytop), (x-dots + 0.4, ytop))
    line((x-c - 1.0, ybot), (x-dots + 0.4, ybot))

    // --- 膜容量 ---
    line((x-c, ytop), (x-c, 4.42))
    line((x-c - 0.62, 4.42), (x-c + 0.62, 4.42))
    line((x-c - 0.62, 3.98), (x-c + 0.62, 3.98))
    line((x-c, 3.98), (x-c, ybot))
    content((x-c + 0.75, 4.2), text(size: label-size)[$C_m$], anchor: "west")

    // --- イオン電流の枝 ---
    ion(x-leak, $overline(g)_"L"$, $E_"L"$, var: false)
    ion(x-na, $overline(g)_"Na" m^2 h$, $E_"Na"$)
    ion(x-k, $overline(g)_"K(DR)" n$, $E_"K"$)
    content((x-dots, 4.2), text(size: label-size + 2pt)[$dots.c$])

    // --- 隣接コンパートメントとの軸方向結合 ---
    line((x-c - 1.0, ytop), (x-c - 1.5, ytop))
    vres(x-c - 2.3, ytop, var: false)
    line((x-c - 3.1, ytop), (x-c - 3.5, ytop))
    circle((x-c - 3.6, ytop), radius: 0.12, fill: white)
    content((x-c - 2.3, ytop + 0.95), text(size: label-size)[$g_(i-1,i)$])
    content((x-c - 3.8, ytop), text(size: label-size)[$V_(i-1)$], anchor: "east")

    line((x-dots + 0.4, ytop), (x-dots + 0.9, ytop))
    vres(x-dots + 1.7, ytop, var: false)
    line((x-dots + 2.5, ytop), (x-dots + 2.9, ytop))
    circle((x-dots + 3.0, ytop), radius: 0.12, fill: white)
    content((x-dots + 1.7, ytop + 0.95), text(size: label-size)[$g_(i,i+1)$])
    content((x-dots + 3.2, ytop), text(size: label-size)[$V_(i+1)$], anchor: "west")

    // --- 外部注入電流 ---
    line(
      (9.6, ytop + 1.6),
      (9.6, ytop + 0.15),
      mark: (end: "stealth", scale: 0.4),
    )
    content(
      (9.75, ytop + 1.65),
      text(size: label-size)[$I_"ext"$],
      anchor: "south-west",
    )

    // --- 膜電位ノードと接地 ---
    content(
      (x-c - 1.0, ytop + 0.25),
      text(size: label-size)[$V_i$ (intracellular)],
      anchor: "south-west",
    )
    let x-gnd = x-c - 1.0
    line((x-gnd, ybot), (x-gnd, -0.6))
    line((x-gnd - 0.5, -0.6), (x-gnd + 0.5, -0.6))
    line((x-gnd - 0.3, -0.85), (x-gnd + 0.3, -0.85))
    line((x-gnd - 0.12, -1.1), (x-gnd + 0.12, -1.1))
    content(
      (x-gnd - 0.7, -0.6),
      text(size: label-size)[extracellular],
      anchor: "east",
    )
  },
)
