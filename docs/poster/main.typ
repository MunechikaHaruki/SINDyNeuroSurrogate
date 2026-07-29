#import "@preview/peace-of-posters:0.5.6" as pop
#import "@preview/typsium:0.3.1": *
#import "circuit.typ": traub_circuit
#import "pipeline.typ": stage_compress, stage_identify, stage_simulate
#import "diagrams.typ": comp-annotated

#set page("a0", margin: (x: 2cm, top: 2cm, bottom: 0.1cm))
#pop.set-poster-layout(pop.layout-a0)
#pop.set-theme(pop.uni-fr)
#set text(size: pop.layout-a0.at("body-size"))
#set text(font: ("New Computer Modern", "Hiragino Kaku Gothic ProN"))
#let box-spacing = 0em
#set columns(gutter: box-spacing)
#set block(spacing: box-spacing)
#pop.update-poster-layout(spacing: box-spacing)

#set text(lang: "en")

#let body-size = 33pt
#let poster-number = "3P-479"

// --- mini box helper ---
#let mini-box(title: "Heading", color: blue, title-size: 35pt, body-inset: 8pt, body) = {
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
        text(fill: white, weight: "bold", size: title-size)[#title],
      ),
      block(
        width: 100%,
        fill: white,
        inset: body-inset,
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
  authors: "⚪︎ Haruki Munechika, Taira Kobayashi",
  institutes: "Grad Sch of Sci and Tech for Innov, Yamaguchi University (f032vbw@yamaguchi-u.ac.jp)",
)

#place(
  top + right,
  dx: -1cm,
  dy: 5cm,
  text(size: 70pt, weight: "bold", fill: white)[#poster-number],
)


#pop.column-box(heading: "Introduction")[
  #set text(size: body-size)
  #grid(
    columns: (1fr, 3.15fr),
    gutter: 1em,
    // ======== 左: 実際の錐体細胞と、そのコンパートメント分割 ========
    [
      #v(0.5em)
      #figure(
        image("pic/ref/pyramidal.png", width: 90%),
        caption: [Guinea pig CA3 pyramidal neuron@pyramidal],
        numbering: none,
        supplement: none,
      )
      #v(2em)
      #figure(
        comp-annotated(w: 18, font: 20pt),
        caption: [Modelled as\ *19 Compartments (comps)* @Traub-1991-ModelCA3HippocampalPyramidal],
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
            numbering: none,
            supplement: none,
          )<circuit>
          // 訳: 各コンパートメント = 膜容量 + 可変イオンコンダクタンス + 隣接との軸方向結合。
          #figure(
            text(size: 28pt)[
              $ C_m (d V) / (d t) = -g_"leak" (V - V_"leak") & - overline(g)_"Na" m^2 h (V - V_"Na") - overline(g)_"K" n (V - V_"K") \
              & + I_"other ionic currents" + I_"ext" $
            ],
            numbering: none,
            supplement: none,
          )
        ],
        // -------- 右: 可変コンダクタンスの中身 (ゲート変数とレート関数) --------
        [
          // 訳: コンダクタンスはゲート変数に依存し、ゲートはレート関数の ODE に従う。
          The conductances depend on *gate variables*, which follow ODEs with rate functions $alpha, beta$:
          #v(1em)
          #figure(
            text(size: 28pt)[
              $
                I_#ce("Na") &= overline(g)_#ce("Na") med m^2 h med (V - E_#ce("Na")) \
                frac(d m, d t) &= alpha_m (V) (1 - m) - beta_m (V) m \
                alpha_m (V) &= 0.32 (13.1 - u) \/ (exp((13.1 - u) \/ 4) - 1) \
                beta_m (V) &= 0.28 (u - 40.1) \/ (exp((u - 40.1) \/ 5) - 1) \
              $
            ],
            kind: "equation",
            numbering: none,
            supplement: none,
          )<eq-gate>
          #v(0.8em)
          // 訳: 1 コンパートメント 11 状態変数 → 19 comp で 209 → 並列シミュレーションでメモリボトルネック。
          $->$ *11 states (10 gates and V) per comp* \
          $->$ *209* states for 19 comps; \
          In large scale network simulations, the number of large gate variables becomes a *memory bottleneck*.
        ],
      )
      #v(0.4em)
      #mini-box(title:"Purpose")[
        Development of a surrogate model of the multi-compartment neuron capable of reproducing the membrane potential response with fewer gate variables.
      ]
    ],
  )
]

#pop.column-box(
  heading: "Methods",
)[
  // 訳: Methods は図と式が多いので本文だけ 1 段小さく (Intro 33pt / Results 24pt の中間)
  #set text(size: 24pt)

  #grid(
    columns: (0.8fr, 0.8fr, 1fr, 1.3fr),
    gutter: 0em,
    row-gutter: 0.4em,
    // ======== ① 刺激を入れ、全 comp の V とゲートの時系列を収集 ========
    [
      // 訳: ① 教師データを集める。
      *#text(blue)[①] Sample the training data*
      #v(0.2em)
      #stage_simulate(unit: 1.1cm, label-size: 20pt)
      #v(0.2em)
      // 訳: Traub 19-comp の soma へランダムパルス列を注入し、19 comp すべての V と 10 ゲートを記録。
      Inject a *random pulse train* at the soma; record $V$ and *6 gates*.
    ],
    // ======== ② 純電位依存ゲート 8 本だけ潜在へ圧縮 (V と Ca サブ系は素通し) ========
    [
      // 訳: ② 電位依存ゲートだけを圧縮する。
      *#text(blue)[②] Compress the gates*
      #v(2em)
      #stage_compress(unit: 1.05cm, label-size: 20pt)
      #v(1.5em)
      // 訳: 純電位依存の 6 ゲートは低次元多様体に乗る → n 次元潜在 z へ (n=5)。V と Ca サブ系 (S,R,Q,ξ) は圧縮しない。
      // 訳: AE 仕様。encoder/decoder それぞれ隠れ層 1 層、損失は MSE ベースの再構成誤差。
      The *6 gates* are compressed to 5 latent variables by an AutoEncoder (encoder and decoder each with one hidden layer).
      #v(0.2em)
    ],
    // ======== ③ [V, z] から潜在の支配方程式を同定 (図は簡略化し、展開式は図の下に置く) ========
    [
      // 訳: ③ 潜在の支配方程式を同定する。
      *#text(blue)[③] Identify ODEs of the latent variables*
      #v(2em)
      #stage_identify(unit: 1.5cm, label-size: 20pt)
      #v(0.2em)
      // 訳: 実際に同定された式の展開 (z1, z2)。基底は 1 項だけ丸で強調していた昨年図の代わりに、ここで具体形を示す。
      #text(size: 20pt)[
        $
          dot(z)_1 &= xi_11 alpha_m (V) + xi_12 beta_m (V) z_1 + dots.c \
          dot(z)_2 &= xi_21 alpha_m (V) + xi_22 beta_m (V)z_1+ dots.c \
          dots.v
        $
      ]
      #v(1.3em)


      Capture the latent variable dynamics with *SINDy*@Champion-2019-DatadrivenDiscoveryCoordinatesGoverning.
      SINDy fits coefficients $Xi$.
      // *SINDy*  fits *only* $dot(bold(z))$; $dot(V)$ keeps *original physics*. Library is *physics-informed*: gates' own $alpha(V), beta(V)$.
    ],
    // ======== ④ 学習済み decoder/SINDy を等価回路の gate 計算へ差し込んでシミュレーション ========
    [
      // 訳: ④ 推論時: decoder-in-the-loop でシミュレーションする。
      *#text(blue)[④] How to apply the surrogate model*
      #image("pic/ref/model.png",width:100%)
      #v(0.5em)

      // Each step: $bold(z) ->$ *decode* $->$ gates feed the equivalent-circuit $dot(V)$.\
      // SINDy updates $bold(z)$ in place of the original gate ODEs.
      The derivative of *$V$* is computed with the *decoded latent variables*.
      Across time steps, *only the latent variables and the membrane potential need to be preserved*.
    ],
  )
]

// tighten figure spacing
#show figure: set block(spacing: 1em)
#show figure: set figure(gap: 0em)

#pop.column-box(heading: "Results and Discussion")[
  #set text(size: 29pt)
  // 掲載は全て同一 run: hybrid / n=5 / AE / traub_sr_physics を traub19 の全 comp へ適用
  // 構成: 2 列。左列 = train_raw + train_preprocessed 縦積み → ④SINDy係数。右列 = ①画像 → ②③。
  #grid(
    columns: (1.3fr, 1fr),
    gutter: 1.5em,
    // ======== 左列: train_raw / train_preprocessed を先頭、以下④ ========
    [
      #grid(
        columns: (1fr,1fr),
        gutter: 1em
      )[
      *Training Data to capture gate dynamics*
      #align(center)[
        #figure(
          image("result/train_raw.png", width: 100%),
          caption: [Raw training trajectories.],
          numbering: none,
          supplement: none,
        )
        #sym.arrow.b compress gates by AutoEncoder
        #figure(
          image("result/train_preprocessed.png", width: 100%),
          caption: [Teaching data to identify the ODEs.],
          numbering: none,
          supplement: none,
        )
      ]
      ][
        *Action Potential reproduction*
        #align(center)[
          #figure(
            image("result/diff.png", width: 100%),
            caption: [20 ms, 3 #sym.mu A/cm#super[2] step: $V$ and the *5 AE latents*.],
            numbering: none,
            supplement: none,
          )
        ]
        #text(size: 25pt)[
          // 訳: いつ起きるか(タイミング)は正確: 潜時誤差0.3ms、AHPタイミング誤差3.1ms。
          - latency err *0.3 ms*, AHP timing gap *3.1 ms*.
          // 訳: どれだけ大きいかは系統的過小評価: ピーク13mV低い(振幅差12.9mV)、立ち上がり/立ち下がり速度差21.0/10.3 mV/ms。一方AHP深さ差0.28mVと静止電位付近は正確。
          - peak *13 mV low* (amplitude gap *12.9 mV*), rise #sym.slash fall rate gap *21.0* #sym.slash *10.3 mV/ms* — yet AHP depth gap is only *0.28 mV*.
        ]
      ]

      #v(0.3em)
      // -------- ④ SINDy 係数 --------
      *Identified latent equations* #h(3em) SINDy coefficients: 79.6% non-zero
      #figure(
          image("result/model.png", width: 100%),
          numbering: none,
          supplement: none,
        )

    ],
    // ======== 右列: ① 画像を先頭、以下②③ を縦に重ねて配置 + 説明 ========
    [
      *Replace the soma compartment with the surrogate model*

      #text(fill:red)[soma compartment (9th comp)] is replaced with the surrogate model.

      #align(center)[
        #figure(
          image("pic/inject_steady_current.png", width: 70%),
          numbering: none,
          supplement: none,
        )]

      #align(center)[
        #figure(
          image("result/compare_stim_site.png", width: 100%),
          caption: [Amplitude sweep: *Top*: Inject to soma. *Bottom*: Inject to dendrite.],
          numbering: none,
          supplement: none,
        )
      ]

      // 訳: 自発発火 (I=0) は再現できないが、I≥2.5 では両注入点でバーストを再現。閾値付近 (2.5) は発火が遅れ、I≥5 で一致。バースト後の静止電位はやや高い。
      - The model failed at spontaneous firing but reproduced *bursts for $I gt.eq 2.5$*: delayed near threshold, and matching spike timing with an elevated post-burst rest for $I gt.eq 5$.

      #align(center)[
        #figure(
          image("pic/inject_periodic_current.png", width: 70%),
          numbering: none,
          supplement: none,
        )]
      #align(center)[
        #figure(
          image("result/traces.png", width: 100%),
          caption: [Frequency sweep: Inject to soma.],
          numbering: none,
          supplement: none,
        )
      ]

    // 訳: 30Hz以上では初期バーストも後続の閾値下応答も一致。10-20Hzでは後続スパイクの時刻がずれる(10Hzで約50ms前倒し)。
    - The response to the periodic current is reproduced with high accuracy, despite a slight phase shift.

      // // 訳: I≥5 で両注入点ともバースト再現、閾値付近は前倒し。
      // - Bursts reproduced at *both sites* for $I gt.eq 5$; fires *too early* near threshold.

      // // 訳: 20 Hz 以上で一致、10 Hz では後続スパイクを落とす。
      // - Matches for $f gt.eq 20$ Hz; *drops later spikes* at 10 Hz.
      // 

    ],
  )
]

// ======== Footer: Conclusion / Code / References / COI (poster 全体の下部) ========
#block(width: 100%, above: 0.5em, below: 1em)[
  #grid(
    columns: (1.5fr, 1fr),
    gutter: 1em,[
    #mini-box(title: "Conclusion", title-size: 30pt, body-inset: 6pt)[
      // #set text(size: 29pt)
      #set par(leading: 0.4em)
      // #set block(spacing: 0.35em)
      // 訳: soma 区画のみ、ゲート変数 6→5 のサロゲートへ置換。
      - Replaced the *soma* compartment only, reducing one gate variable gate variable.
      // 訳: 学習した刺激条件では波形を良好に再現。
      - Waveforms are *accurately reproduced* within the trained stimulus condition.
      // 訳: 学習データ外のダイナミクスは再現できず。
      - Dynamics *outside the training data* like spotaneous firing are *not reproduced*.


    ]
    #v(0.5em)
    #set par(leading: 0.4em)
    *Future work*
    // 訳: 元の方程式の構造をより強く捉える次元圧縮・ODE 同定手法を試す。
    - Explore dimensionality reduction and ODE identification methods that capture the *structure of the original equations* more strongly.
    // 訳: ゲート変数をさらに減らしつつ、全コンパートメントを高精度に置換する。
    - Replace *all compartments* accurately, with the gate variables reduced further.
    ],
    [
      #text(size: 20pt)[*Code* — #link("https://github.com/MunechikaHaruki/SINDyNeuroSurrogate")[github.com/MunechikaHaruki/SINDyNeuroSurrogate]]
      #v(0.5em)
      #show bibliography: set text(size: 20pt)
      #bibliography("bibliography.bib", title: none)
      #v(0.3em)
      #block(
        width: 100%,
        stroke: 1pt + black,
        radius: 4pt,
        inset: 6pt,
        text(size: 14pt)[*Conflicts of Interest* — The authors declare no conflicts of interest regarding this manuscript.],
      )
    ],
  )
]
