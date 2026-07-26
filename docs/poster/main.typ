#import "@preview/peace-of-posters:0.5.6" as pop
#import "@preview/typsium:0.3.1": *
#import "circuit.typ": traub_circuit
#import "pipeline.typ": stage_compress, stage_identify, stage_simulate

#set page("a0", margin: 2cm)
#pop.set-poster-layout(pop.layout-a0)
#pop.set-theme(pop.uni-fr)
#set text(size: pop.layout-a0.at("body-size"))
#set text(font: ("New Computer Modern", "Hiragino Kaku Gothic ProN"))
#let box-spacing = 0.5em
#set columns(gutter: box-spacing)
#set block(spacing: box-spacing)
#pop.update-poster-layout(spacing: box-spacing)

#set text(lang: "en")

#let body-size = 33pt

// --- mini box helper ---
#let mini-box(title: "Heading", color: blue, body) = {
  block(
    width: 100%,
    stroke: 1pt + color,
    radius: 4pt,
    clip: true,
    stack(
      dir: ttb,
      block(
        width: 100%,
        fill: color,
        inset: 6pt,
        text(fill: white, weight: "bold", size: 35pt)[#title],
      ),
      block(
        width: 100%,
        fill: white,
        inset: 8pt,
        body,
      ),
    ),
  )
}


#pop.title-box(
  // 訳: ニューロンシミュレーションの計算コスト削減のためのサロゲートモデル
  text(size: 60pt)[
    A Surrogate Model for Reducing the Computational Cost of Neuron Simulations #v(-3em)
  ],
  authors: "Haruki Munechika",
  institutes: "",
)


#pop.column-box(heading: "Introduction")[
  #set text(size: body-size)
  #grid(
    columns: (1fr, 3.15fr),
    gutter: 1em,
    // ======== 左: 実際の錐体細胞と、そのコンパートメント分割 ========
    [
      #figure(
        image("pic/ref/pyramidal_.png", width: 100%),
        caption: [CA3 pyramidal neuron @noauthor__2012],
        numbering: none,
        supplement: none,
      )
      #v(0.4em)
      #figure(
        image("pic/ref/traub_comp.png", width: 100%),
        caption: [Modelled as *19 compartments* @Traub-1991-ModelCA3HippocampalPyramidal],
        numbering: none,
        supplement: none,
      )<comp>
    ],
    // ======== 中央+右: 等価回路 / 可変コンダクタンスの中身、その下に Goal を跨がせる ========
    [
      #grid(
        columns: (1.7fr, 1.45fr),
        gutter: 1em,
        // -------- 中央: 1 コンパートメントの等価回路 --------
        [
          #figure(
            traub_circuit(unit: 1.02cm, label-size: 24pt, stroke-w: 1.6pt),
            caption: [Equivalent circuit of compartment $i$],
            numbering: none,
            supplement: none,
          )<circuit>
          // 訳: 各コンパートメント = 膜容量 + 可変イオンコンダクタンス + 隣接との軸方向結合。
          Each compartment: capacitance, *variable* ionic conductances, axial coupling $g_(i, i plus.minus 1)$.
        ],
        // -------- 右: 可変コンダクタンスの中身 (ゲート変数とレート関数) --------
        [
          // 訳: コンダクタンスはゲート変数に依存し、ゲートはレート関数の ODE に従う。
          Conductances depend on *gate variables*, ODEs with rate functions $alpha, beta$:
          #figure(
            text(size: 28pt)[
              $
                I_#ce("Na") &= overline(g)_#ce("Na") med m^2 h med (V - E_#ce("Na")) \
                frac(d m, d t) &= alpha_m (V) (1 - m) - beta_m (V) m \
                alpha_m (V) &= 0.32 (13.1 - u) \/ (exp((13.1 - u) \/ 4) - 1)
              $
            ],
            caption: [#ce("Na") current, its gate $m$, and one rate function ($u = V - E_L$)],
            kind: "equation",
            numbering: none,
            supplement: none,
          )<eq-gate>
          #v(-0.3em)
          // 訳: 1 コンパートメント 11 状態変数 → 19 comp で 209 → 並列シミュレーションでメモリボトルネック。
          $->$ *11 states per compartment* $->$ *209* for 19; at brain scale the gates become a *memory bottleneck*.
        ],
      )
      #v(0.4em)
      // 訳 (Goal): より少ない状態変数で膜電位応答を再現するサロゲートモデルを構築する。
      #mini-box(title: "Goal")[
        Reproduce the membrane-potential response with *fewer state variables*.
      ]
    ],
  )
]

#pop.column-box(
  heading: "Methods",
)[
  // 訳: Methods は図と式が多いので本文だけ 1 段小さく (Intro 33pt / Results 24pt の中間)
  #set text(size: 29pt)

  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 0.8em,
    // ======== ① 刺激を入れ、全 comp の V とゲートの時系列を収集 ========
    [
      // 訳: ① 教師データを集める。
      *#text(blue)[①] Collect the training data*
      #v(0.2em)
      #stage_simulate(unit: 1.62cm, label-size: 24pt)
      #v(0.2em)
      // 訳: Traub 19-comp の soma へランダムパルス列を注入し、19 comp すべての V と 10 ゲートを記録。
      Inject a *random pulse train* at the soma; record $V$ and *10 gates* for *all 19 compartments*.
    ],
    // ======== ② 純電位依存ゲート 8 本だけ潜在へ圧縮 (V と Ca サブ系は素通し) ========
    [
      // 訳: ② 電位依存ゲートだけを圧縮する。
      *#text(blue)[②] Compress _only_ the gates*
      #v(0.2em)
      #stage_compress(unit: 1.57cm, label-size: 24pt)
      #v(0.2em)
      // 訳: 純電位依存の 6 ゲートは低次元多様体に乗る → n 次元潜在 z へ (n=5)。V と Ca サブ系 (S,R,Q,ξ) は圧縮しない。
      *6 voltage-dependent gates* ride a *low-dimensional manifold* $->$ $bold(z) in RR^n$, $n=5$. #text(blue)[$V$] and #ce("Ca^2+") *stay uncompressed*.
    ],
    // ======== ③ [V, z] から潜在の支配方程式を同定 ========
    [
      // 訳: ③ 潜在の支配方程式を同定する。
      *#text(blue)[③] Identify the latent ODEs*
      #v(0.2em)
      #stage_identify(unit: 1.58cm, label-size: 24pt)
      #v(0.2em)
      // 訳: 潜在だけスパース同定、dV/dt と Ca サブ系は元の物理式のまま。基底はゲート自身のレート関数 α, β (昨年の 41 項汎用ライブラリを置換)。
      *SINDy* @Champion-2019-DatadrivenDiscoveryCoordinatesGoverning fits *only* $dot(bold(z))$; $dot(V)$ and #ce("Ca^2+") keep *original physics*. Library is *physics-informed*: gates' own $alpha(V), beta(V)$, not a *41-term* generic one.
    ],
  )
]

// tighten figure spacing
#show figure: set block(spacing: 1em)
#show figure: set figure(gap: 0em)

#pop.column-box(heading: "Results and Discussion")[
  #set text(size: 22pt)
  // 掲載は全て同一 run: hybrid / n=5 / AE / traub_sr_physics を traub19 の全 comp へ適用
  // 外側 2 列 = (左+中央 / 右)。model.png が横長なので左+中央に跨がせる (Intro の Goal と同じ手)。
  #grid(
    columns: (2.5fr, 0.8fr),
    gutter: 1em,
    // ======== 左+中央: ①② ③ を 2 列に並べ、その下へ ④ を跨がせる ========
    [
      #grid(
        columns: (1fr, 1.3fr),
        gutter: 1em,
        // -------- 左列: 単発 AP の再現 --------
        [
          *① Single action potential*
          #figure(
            image("result/diff.png", width: 100%),
            caption: [20 ms, 3 #sym.mu A/cm#super[2] step: $V$ and the *5 AE latents*. Original (blue) / surrogate (red).],
            numbering: none,
            supplement: none,
          )
          #mini-box(title: "Waveform match", color: rgb("#2a7f2a"))[
            #set text(size: 20pt)
            // 訳: RMSE 7.3 mV、AP 形状相関 0.999、スパイク数一致、潜時誤差 0.3 ms
            - RMSE *7.3 mV*, shape corr. *0.999*
            - spike *1 #sym.arrow.r 1*, latency err *0.3 ms*
          ]
          // 訳: 残る誤差はピーク電圧 13 mV 低い。
          - *Gap*: peak *13 mV low*.
        ],
        // -------- 中央列: 学習外の刺激への汎化 (②③) --------
        [
          *② New stimulus site*
          #figure(
            image("result/compare_stim_site.png", width: 100%),
            caption: [Amplitude sweep. *Top*: soma, as trained. *Bottom*: dendrite, *unseen*.],
            numbering: none,
            supplement: none,
          )
          // 訳: I≥5 で両注入点ともバースト再現、閾値付近は前倒し。
          - Bursts reproduced at *both sites* for $I gt.eq 5$; fires *too early* near threshold.
          #v(0.25em)
          *③ Unseen periodic drive*
          #figure(
            image("result/traces.png", width: 100%),
            caption: [Pulse train, 10–50 Hz — *outside training*.],
            numbering: none,
            supplement: none,
          )
          // 訳: 20 Hz 以上で一致、10 Hz では後続スパイクを落とす。
          - Matches for $f gt.eq 20$ Hz; *drops later spikes* at 10 Hz.
        ],
      )
      #v(0.3em)
      // -------- ④ 同定された潜在方程式 (横長なので左+中央に跨る) --------
      *④ Identified latent equations*
      #figure(
        image("result/model.png", width: 100%),
        caption: [$xi$ over the physics-informed library.],
        numbering: none,
        supplement: none,
      )
      // 訳: 79.6% が非ゼロ = スパースでない → コスト増。
      - *79.6% non-zero* — accurate but *not sparse*: cost *rises* (exp $19 -> 121$).
    ],
    // ======== 右列: 上は空けたまま、下に結論と参照 ========
    [
      #mini-box(title: "Conclusion")[
        #set text(size: 21pt)
        // 訳: Traub 19-comp 全置換で発散せず、AP 形状相関 0.999。
        - *All 19 compartments* replaced, no divergence; AP *shape corr. 0.999*.
        // 訳: 学習外の刺激位置・周波数へ転移 → 部品として再利用可。
        - Transfers to an *unseen site and frequency*.
        // 訳: 圧縮は 6→5 止まりで ξ が密。
        - Compression stops at *6 #sym.arrow.r 5* and $xi$ stays dense.
        // 訳: 今後: ξ のスパース化と潜在次元削減。
        $->$ *Future*: sparsify $xi$, push $n$ down.
      ]
      #v(0.3em)
      #text(size: 21pt)[*Code* — #link("https://github.com/MunechikaHaruki/SINDyNeuroSurrogate")]
      #show bibliography: set text(size: 18pt)
      #bibliography("bibliography.bib", title: none)
    ],
  )
]
