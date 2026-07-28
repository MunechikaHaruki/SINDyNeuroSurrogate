#import "@preview/cetz:0.5.2"

// Introduction: traub_comp 画像に basal/soma/apical dendrite を指す矢印を重ねる
#let comp-annotated(w: 18, font: 20pt) = {
  let h = 332 / 1230 * w
  cetz.canvas({
    import cetz.draw: *
    // 画像 1230x332px を幅 w へ縮尺 (高さ 332/1230*w)
    content((w / 2, h / 2), image("pic/ref/traub_comp.png", width: w * 1cm))
    // 訳: comp1-8 = basal dendrite (左), comp9 = soma (中央), comp10-19 = apical dendrite (右)
    // px→cetz座標: x_cetz = px/1230*w, y は画像上端基準で下向き正 → cetz は上向き正なので h - px/332*h
    let px-to-x(px) = px / 1230 * w
    let py-to-y(py) = h - py / 332 * h

    // basal dendrite (comp1-8, 左側) を指す矢印
    // line((px-to-x(50), py-to-y(-20)), (px-to-x(50), py-to-y(20)), mark: (end: ">"), stroke: 1.5pt + red)
    content((px-to-x(50), py-to-y(-40)), text(size: font, fill: red)[dendrite])

    // soma (comp9) を指す矢印
    line((px-to-x(457), py-to-y(-20)), (px-to-x(457), py-to-y(20)), mark: (end: ">"), stroke: 1.5pt + blue)
    content((px-to-x(457), py-to-y(-40)), text(size: font, fill: blue)[soma])

    // apical dendrite (comp10-19, 右側) を指す矢印
    // line((px-to-x(1030), py-to-y(-20)), (px-to-x(1030), py-to-y(20)), mark: (end: ">"), stroke: 1.5pt + red)
    content((px-to-x(1030), py-to-y(-40)), text(size: font, fill: red)[axon])
  })
}

