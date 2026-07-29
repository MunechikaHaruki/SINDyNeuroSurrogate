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
  institutes: "Graduate School of Sciences and Technology for Innovation, Yamaguchi University (f032vbw@yamaguchi-u.ac.jp)",
  institutes-size: 32pt,
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
        image("pic/ref/pyramidal.png", width: 95%),
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
              $ C_m (d v) / (d t) = -g_"leak" (v - E_"leak") & - overline(g)_"Na" m^2 h (v - E_"Na") - overline(g)_"K" n (v - E_"K") \
              & + I_"other ionic currents" + I_"ext" $
            ],
            numbering: none,
            supplement: none,
          )
        ],
        // -------- 右: 可変コンダクタンスの中身 (ゲート変数とレート関数) --------
        [
          // 訳: コンダクタンスはゲート変数に依存し、ゲートはレート関数の ODE に従う。
          The conductances depend on *gate variables*, which follow ODEs with the rate functions $alpha, beta$:
          #v(1em)
          #figure(
            text(size: 28pt)[
              $
                I_#ce("Na") &= overline(g)_#ce("Na") med m^2 h med (v - E_#ce("Na")) \
                frac(d m, d t) &= alpha_m (v) (1 - m) - beta_m (v) m \
                alpha_m (v) &= 0.32 (13.1 - v) \/ (exp((13.1 - v) \/ 4) - 1) \
                beta_m (v) &= 0.28 (v - 40.1) \/ (exp((v - 40.1) \/ 5) - 1) \
              $
            ],
            kind: "equation",
            numbering: none,
            supplement: none,
          )
          #v(0.8em)
          // 訳: 1 コンパートメント 11 状態変数 → 19 comp で 209 → 並列シミュレーションでメモリボトルネック。
          $->$ *10 gates and the potential $v$ per comp* \
          $->$ *190 gates* for the 19 comps; \
          In large scale network simulations, the *large number of gate variables* becomes a *memory bottleneck*.
        ],
      )
      #v(0.4em)
      #mini-box(title:"Purpose")[
        Development of a surrogate model for the multi-compartment model, capable of reproducing the membrane potential response with fewer gate variables.
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
      #v(0.5em)
      #stage_simulate(unit: 1.1cm, label-size: 20pt)
      #v(0.2em)
      // 訳: Traub 19-comp の soma へランダムパルス列を注入し、V と純電位依存の 6 ゲートを記録 (Ca 依存系は対象外)。
      Inject a *random pulse train* at the soma; record $v(t)$ and the *6 gates* $m(t), n(t), dots$.
      #v(0.2em)
      #text(size: 20pt)[
        (The remaining gates, driven by #ce("Ca^2+") dynamics, are left untouched.)
      ]
    ],
    // ======== ② 純電位依存ゲート 6 本だけ潜在へ圧縮 (V と Ca サブ系は素通し) ========
    [
      // 訳: ② 電位依存ゲートだけを圧縮する。
      *#text(blue)[②] Compress the 6 gates*
      #v(3em)
      #stage_compress(unit: 1.05cm, label-size: 20pt)
      #v(4em)
      // 訳: 純電位依存の 6 ゲートは低次元多様体に乗る → n 次元潜在 z へ (n=5)。V と Ca サブ系 (S,R,Q,ξ) は圧縮しない。
      // 訳: AE 仕様。encoder/decoder それぞれ隠れ層 1 層、損失は MSE ベースの再構成誤差。
      The *6 gates* $m(t), n(t), dots$ are compressed to the *5 latent variables* $z_1 (t), dots, z_5 (t)$ by the encoder of an AutoEncoder (one hidden layer). $v(t)$ is *not* compressed.
      #v(0.2em)
    ],
    // ======== ③ [V, z] から潜在の支配方程式を同定 (図は簡略化し、展開式は図の下に置く) ========
    [
      // 訳: ③ 潜在の支配方程式を同定する。
      *#text(blue)[③] Identify ODEs of the latent variables $z(t)$*
      #v(2em)
      #stage_identify(unit: 1.5cm, label-size: 20pt)
      #v(0.2em)
      // 訳: 実際に同定された式の展開 (z1, z2)。基底は 1 項だけ丸で強調していた昨年図の代わりに、ここで具体形を示す。
      #text(size: 20pt)[
        $
          (d z_1) / (d t) &= xi_11 alpha_m (v) + xi_12 beta_m (v) z_1 + dots.c \
          (d z_2) / (d t) &= xi_21 alpha_m (v) + xi_22 beta_m (v) z_1 + dots.c \
          dots.v
        $
      ]
      #v(1.0em)


      To write the ODEs $(d z) / (d t)$ of the time series $z(t)$ as a linear sum of *basis functions*, the machine learning model *SINDy*@Champion-2019-DatadrivenDiscoveryCoordinatesGoverning identifies the coefficients $xi_(i j)$.
      #v(0.2em)
      #text(size: 20pt)[
        The basis functions are the rate functions $alpha(v), beta(v)$ of the gate variables, and their products with $z_1 (t), dots, z_5 (t)$.
      ]
    ],
    // ======== ④ 学習済み decoder/SINDy を等価回路の gate 計算へ差し込んでシミュレーション ========
    [
      // 訳: ④ 推論時: decoder-in-the-loop でシミュレーションする。
      *#text(blue)[④] How to calculate the $v(t)$ with the compressed $z(t)$*
      #image("pic/ref/model.png",width:100%)
      #v(0.2em)
      $v(t + Delta t)$ is calculated using the gate variables ($m(t), n(t), dots$) decoded by the *decoder* of the AutoEncoder from the latent variables ($z_1 (t), z_2 (t), dots$).
    ],
  )
]

// tighten figure spacing
#show figure: set block(spacing: 1em)
#show figure: set figure(gap: 0em)

#pop.column-box(heading: "Results and Discussion")[
  #set text(size: 25pt)
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
      *Training data to capture the gate dynamics*
      #align(center)[
        #figure(
          image("result/train_raw.png", width: 100%),
          caption: none,
          numbering: none,
          supplement: none,
        )
        #sym.arrow.b *The 6 gates* are compressed into *5 latent variables*
        #figure(
          image("result/train_preprocessed.png", width: 100%),
          caption: none,
          numbering: none,
          supplement: none,
        )
      ]
      ][
        *Reproducibility of a somatic surrogate model*
        #align(center)[
          #figure(
            image("result/diff.png", width: 100%),
            caption: none,
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
      *Identified equations of the latent variables*
      #figure(
          image("result/model.png", width: 100%),
          numbering: none,
          supplement: none,
        )
      #v(0.2em)
      79.6 % of the coefficients of $(d z) / (d t)$ are non-zero.

    ],
    // ======== 右列: ① 画像を先頭、以下②③ を縦に重ねて配置 + 説明 ========
    [
      *Reproducibility of a multi-compartment model in which the soma compartment was replaced by the somatic surrogate model*

      #text(fill:red)[soma compartment (9th comp)] is replaced with the surrogate model.

      #align(center)[
        #figure(
          image("pic/inject_steady_current.png", width: 70%),
          numbering: none,
          supplement: none,
        )]

      #align(center)[
        #figure(
          image("result/compare_stim_site.png", width: 95%),
          caption: [*Top*: Inject to soma. *Bottom*: Inject to dendrite.],
          numbering: none,
          supplement: none,
        )
      ]

      // 訳: 定電流入力に対して膜電位をよく再現。自発活動 (0 µA/cm²) は示さず、学習データに自発活動が含まれないことが原因かもしれない。
      - The surrogate model *reproduced the membrane potential well* for the constant currents.
      - It did *not show the spontaneous firing* of the original model (left). The training data may not contain the spontaneous firing.

      #align(center)[
        #figure(
          image("pic/inject_periodic_current.png", width: 70%),
          numbering: none,
          supplement: none,
        )]
      #align(center)[
        #figure(
          image("result/traces.png", width: 95%),
          caption: none,
          numbering: none,
          supplement: none,
        )
      ]

    // 訳: パルス入力も同様。30 Hz 以上ではよく再現、10 Hz では自発活動を再現できず約 10 ms 早く発火。
    - For the pulse currents, the surrogate model *reproduced the membrane potential well* for $f gt.eq 30$ Hz.
    - At 10 Hz, it fired about 10 ms earlier than the original model, and it did not show the spontaneous firing between the pulses.

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
      #set text(size: 24pt)
      #set par(leading: 0.3em)
      // #set block(spacing: 0.35em)
      // 訳: soma のみをゲート変数 6→5 のサロゲートに置換してシミュレーション。再現性は良好だが、学習データに無いダイナミクスは再現できず。
      - We simulated the multi-compartment model in which only the *soma compartment* was replaced by the surrogate model with *fewer gate variables (6 $->$ 5)*.
      - The surrogate model *reproduced the membrane potential well*, but it did not reproduce the dynamics that were *not contained in the training data*.

    ]
    #v(0.5em)
    #set par(leading: 0.3em)
    #set text(size: 24pt)
    *Future work*
    // 訳: ゲート変数をさらに減らした場合の再現性を確かめる。
    - Check the reproducibility when the number of gate variables is reduced further.
    // 訳: soma 以外も置換した full surrogate model の再現性を確かめる。
    - Check the reproducibility of the *full surrogate model*, in which the compartments other than the soma are also replaced.
    ],
    [
      #text(size: 18pt)[*Code* — #link("https://github.com/MunechikaHaruki/SINDyNeuroSurrogate")[github.com/MunechikaHaruki/SINDyNeuroSurrogate]]
      #v(0.5em)
      #show bibliography: set text(size: 15pt)
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
