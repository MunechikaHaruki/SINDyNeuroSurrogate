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
        caption: [Modeled as\ *19 compartments (comps)* @Traub-1991-ModelCA3HippocampalPyramidal],
        numbering: none,
        supplement: none,
      )
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
          )
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
                alpha_m (V) &= 0.32 (13.1 - V) \/ (exp((13.1 - V) \/ 4) - 1) \
                beta_m (V) &= 0.28 (V - 40.1) \/ (exp((V - 40.1) \/ 5) - 1) \
              $
            ],
            kind: "equation",
            numbering: none,
            supplement: none,
          )
          #v(0.8em)
          // 訳: 1 コンパートメント 11 状態変数 → 19 comp で 209 → 並列シミュレーションでメモリボトルネック。
          $->$ *11 states (10 gates and V) per comp* \
          $->$ *209* states for 19 comps; \
          In large scale network simulations, the number of gate variables becomes a *memory bottleneck*.
        ],
      )
      #v(0.4em)
      #mini-box(title:"Purpose")[
        Development of a multi-compartment neuron surrogate model capable of reproducing the membrane potential response with fewer gate variables.
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
    gutter: 0.5em,
    row-gutter: 0.4em,
    // ======== ① 刺激を入れ、全 comp の V とゲートの時系列を収集 ========
    [
      // 訳: ① 教師データを集める。
      *#text(blue)[①] Sample training data*
      #v(0.2em)
      #stage_simulate(unit: 1.1cm, label-size: 20pt)
      #v(0.2em)
      // 訳: Traub 19-comp の soma へランダムパルス列を注入し、V と純電位依存の 6 ゲートを記録 (Ca 依存系は対象外)。
      Inject a *random pulse train* at the soma; record $V$ and *6 gates*.
      #v(0.2em)
      #text(size: 20pt)[
        (The remaining gates, driven by #ce("Ca^2+") dynamics, are left untouched.)
      ]
    ],
    // ======== ② 純電位依存ゲート 6 本だけ潜在へ圧縮 (V と Ca サブ系は素通し) ========
    [
      // 訳: ② 電位依存ゲートだけを圧縮する。
      *#text(blue)[②] Compress the gates*
      #v(2em)
      #stage_compress(unit: 1.05cm, label-size: 20pt)
      #v(3.8em)
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
      #v(1.8em)


      Capture the latent variable dynamics with *SINDy*@Champion-2019-DatadrivenDiscoveryCoordinatesGoverning.
      SINDy fits coefficients $Xi$.
      #v(0.2em)
      #text(size: 20pt)[
        The library is *physics-informed*: it is built from the gates'  $alpha(V), beta(V)$.
      ]
    ],
    // ======== ④ 学習済み decoder/SINDy を等価回路の gate 計算へ差し込んでシミュレーション ========
    [
      // 訳: ④ 推論時: decoder-in-the-loop でシミュレーションする。
      *#text(blue)[④] How to apply the surrogate model*
      #image("pic/ref/model.png",width:100%)
      #v(0.5em)

      // Each step: $bold(z) ->$ *decode* $->$ gates feed the equivalent-circuit $dot(V)$.\
      // SINDy updates $bold(z)$ in place of the original gate ODEs.
      The derivative of *$V$* is computed with the *decoded gates*.
      Across time steps, *only the latent variables and the membrane potential need to be stored*.
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
          caption: [Training data for ODE identification.],
          numbering: none,
          supplement: none,
        )
      ]
      ][
        *Action Potential reproduction*
        #align(center)[
          #figure(
            image("result/diff.png", width: 100%),
            caption: [20 ms, 3 #sym.mu#h(0em)A/cm#super[2] step: $V$ and the *5 AE latents*.],
            numbering: none,
            supplement: none,
          )
        ]
        #text(size: 25pt)[
          // 訳: いつ起きるか(タイミング)は正確: 潜時誤差0.3ms、AHPタイミング誤差3.1ms。
          - spike latency error *0.3 ms*, AHP timing gap *3.1 ms*.
          // 訳: どれだけ大きいかは系統的過小評価: ピーク13mV低い(振幅差12.9mV)、立ち上がり/立ち下がり速度差21.0/10.3 mV/ms。一方AHP深さ差0.28mVと静止電位付近は正確。
          - peak *13 mV low* (amplitude gap *12.9 mV*), rise #sym.slash fall rate gap *21.0* #sym.slash *10.3 mV/ms* — yet AHP depth gap is only *0.28 mV*.
        ]
      ]

      #v(1em)
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

      // 訳: 自発発火は再現できず。I≥2.5 でバースト出現、閾値付近では遅れるが I≥5 では時刻良好。バースト後の静止電位は高すぎる。
      - Spontaneous firing is *not reproduced*. *Bursts appear for $I gt.eq 2.5$*: delayed near threshold, but well timed for $I gt.eq 5$. The post-burst resting potential is *more depolarized* than the original.

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

    // 訳: 30Hz以上ではモデルが応答をよく再現。10Hzでは約10ms早く発火し、静止電位が高すぎる。
    - For $f gt.eq 30$ Hz, the model reproduces the response well. At 10 Hz, it fires about 10 ms early.

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
      #set text(size: 29pt)
      #set par(leading: 0.4em)
      // #set block(spacing: 0.35em)
      // 訳: サロゲートは soma 区画のみを置換 (ゲート変数 6→5)、マルチコンパートメントニューロンのサロゲートモデルへの第一歩として。
      - The surrogate replaces the *soma* compartment only (11 $->$ 10 states), as a first step toward a multi-compartment neuron surrogate model.
      // 訳: 学習に用いた刺激条件下では波形を良好に再現。
      - Waveforms are *well reproduced* under the training stimulus conditions.
      // 訳: 自発発火など、学習データ外のダイナミクスは再現できず。
      - Dynamics *outside the training data*, such as spontaneous firing, are *not reproduced*.


    ]
    #v(0.5em)
    #set par(leading: 0.4em)
    #set text(size: 29pt)
    *Future work*
    // 訳: 元の方程式の構造をより強く捉える次元圧縮・ODE 同定手法を試す。
    - Explore dimensionality reduction and ODE identification methods that capture the *structure of the original equations* more strongly.
    // 訳: ゲート変数をさらに削減しつつ、全コンパートメントを正確に置換。
    - Replace *all compartments* accurately, with further reduction of gate variables.
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
