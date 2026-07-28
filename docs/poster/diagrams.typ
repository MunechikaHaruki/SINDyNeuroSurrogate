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
    line((px-to-x(50), py-to-y(-20)), (px-to-x(50), py-to-y(20)), mark: (end: ">"), stroke: 1.5pt + red)
    content((px-to-x(50), py-to-y(-40)), text(size: font, fill: red)[basal dend])

    // soma (comp9) を指す矢印
    line((px-to-x(457), py-to-y(-20)), (px-to-x(457), py-to-y(20)), mark: (end: ">"), stroke: 1.5pt + blue)
    content((px-to-x(457), py-to-y(-40)), text(size: font, fill: blue)[soma])

    // apical dendrite (comp10-19, 右側) を指す矢印
    line((px-to-x(1030), py-to-y(-20)), (px-to-x(1030), py-to-y(20)), mark: (end: ">"), stroke: 1.5pt + red)
    content((px-to-x(1030), py-to-y(-40)), text(size: font, fill: red)[apical dend])
  })
}

// Results: 刺激部位と置換部位の模式図 (dendrite - soma - axon + steady/pulse current + soma置換)
#let stim-replace-diagram(w: 8, h: 1.7, font: 33pt) = {
  cetz.canvas({
    import cetz.draw: *

    // dendrite - soma - axon の3コンパートメントを横に連結
    rect((0, 0), (w, h), name: "dendrite")
    rect((w, 0), (2 * w, h), name: "soma")
    rect((2 * w, 0), (3 * w, h), name: "axon")
    content((w / 2, h / 2), text(size: font)[dendrite])
    content((3 * w / 2, h / 2), text(size: font)[soma])
    content((5 * w / 2, h / 2), text(size: font)[axon])

    // steady current: 一定振幅の矩形波形 → dendrite へ注入
    let wf-y0 = h + 1.7
    let wf-h = 0.5
    let wf-x0 = w / 2 - 0.5
    line(
      (wf-x0, wf-y0), (wf-x0, wf-y0 + wf-h),
      (wf-x0 + 1, wf-y0 + wf-h),
      stroke: 1.5pt + blue,
    )
    line((w / 2, wf-y0 - 0.2), (w / 2, h), mark: (end: ">"), stroke: 1.5pt + blue)
    content((w / 2, wf-y0 + wf-h + 0.35), anchor: "east", text(size: font, fill: blue)[steady current])

    // pulse current: 矩形パルス列 → soma へ注入
    let px0 = 3 * w / 2 - 0.6
    let pw = 0.25
    let gap = 0.2
    line(
      (px0, wf-y0),
      (px0, wf-y0 + wf-h), (px0 + pw, wf-y0 + wf-h), (px0 + pw, wf-y0),
      (px0 + pw + gap, wf-y0),
      (px0 + pw + gap, wf-y0 + wf-h), (px0 + 2 * pw + gap, wf-y0 + wf-h), (px0 + 2 * pw + gap, wf-y0),
      (px0 + 2 * pw + 2 * gap, wf-y0),
      (px0 + 2 * pw + 2 * gap, wf-y0 + wf-h), (px0 + 3 * pw + 2 * gap, wf-y0 + wf-h), (px0 + 3 * pw + 2 * gap, wf-y0),
      stroke: 1.5pt + red,
    )
    line((3 * w / 2, wf-y0 - 0.2), (3 * w / 2, h), mark: (end: ">"), stroke: 1.5pt + red)
    content((3 * w / 2+1.5, wf-y0 + wf-h + 0.35), anchor: "west", text(size: font, fill: red)[pulse current])

    // soma を surrogate で置換することを示す注記 (右へずらしキャプションと重ならないようにする)
    line((3 * w / 2, -0.2), (3 * w / 2, -1.8), mark: (start: ">"), stroke: 1.5pt + eastern)
    content((3 * w / 2 + 1.3, -2.8), text(size: font, fill: eastern)[replace soma / surrogate])
  })
}
