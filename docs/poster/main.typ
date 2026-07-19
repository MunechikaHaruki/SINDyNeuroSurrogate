#import "@preview/peace-of-posters:0.5.6" as pop
#import "@preview/typsium:0.3.1": *

#set page("a0", margin: 2cm)
#pop.set-poster-layout(pop.layout-a0)
#pop.set-theme(pop.uni-fr)
#set text(size: pop.layout-a0.at("body-size"))
#set text(font: ("New Computer Modern", "Hiragino Kaku Gothic ProN"))
#let box-spacing = 1em
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
    columns: (1.5fr, 1fr),
    gutter: 1em,
    [
      // 訳: マルチコンパートメントモデルはニューロンの空間形態を連結コンパートメントで表現し、神経活動を詳細に再現する。
      - Multi-compartment models represent the spatial morphology of a neuron as connected compartments, reproducing neuronal activity in fine detail.
      // 訳: 各コンパートメントは Hodgkin–Huxley (HH) モデルのように、ゲート変数 (イオンチャネル開確率) を含む微分方程式で記述される。
      - Each compartment is described by differential equations carrying *gate variables* (ion-channel open probabilities), as in the Hodgkin–Huxley (HH) model.
        #figure(
          $
            C_m frac(d V, d t) = -overline(g)_L (V - E_L) - overline(g)_#ce("Na") m^3h (V - E_#ce("Na") ) - overline(g)_#ce("K") n^4 (V - E_#ce("K")) + I_"ext" \
          $,
          caption: [HH membrane-potential equation ($m, h, n$: gate variables)],
          kind: "equation",
          numbering: none,
          supplement: none,
        ) <eq-hh>
      // 訳: 多数のマルチコンパートメントモデルを並列にシミュレーションする (例: 全脳規模シミュレーション) と、ゲート変数がメモリボトルネックを生む。
      $->$ When many multi-compartment models are simulated in parallel (e.g. brain-scale simulation), the gate variables cause a *memory bottleneck*.
      #v(0.3em)
      // 訳 (Goal): より少ない状態変数でマルチコンパートメントニューロンの膜電位応答を再現するサロゲートモデルを構築する。
      #mini-box(title: "Goal")[
        Build a surrogate model that reproduces the membrane-potential response of a multi-compartment neuron with *fewer state variables*.
      ]

    ],
    [
      #grid(
        columns: (1fr, 1fr),
        gutter: 1em,
        [
          #figure(
            image("pic/ref/pyramidal_.png", width: 60%),
            caption: [Pyramidal neuron @noauthor__2012],
            numbering: none,
            supplement: none
          )
          #figure(
            image("pic/ref/traub_comp.png", width: 60%),
            caption: none,
            numbering: none,
            supplement: none
          )<comp>

          #figure(
            image("pic/ref/traub_withcap.png",width: 70%),
            caption: [Spatial representation and simulation of a multi-compartment model @Traub-1991-ModelCA3HippocampalPyramidal],
            numbering: none,
            supplement: none
          )<diff>
        ],
        [
          #figure(
            image("pic/ref/yamazaki.png"),
            caption: [HH membrane-potential response @山﨑匡-2021-はじめて],
            numbering: none,
            supplement: none
          )<brain-yamazaki>
        ],
      )
    ],
  )
]

#pop.column-box(
  heading: "Methods",
)[
  #set text(size: body-size)

  #grid(
    columns: (1fr, 1fr, 0.5fr),
    gutter: 0.3em,
    [
      // 訳: 2軸サロゲートフレームワーク
      *Two-axis surrogate framework*
      // 訳: 対象モデルをシミュレーションし、膜電位 V とゲート変数の時系列を収集する。
      + Simulate the target model; collect time series of membrane potential $V$ and gate variables.
      // 訳: preprocessor — PCA (線形) または autoencoder (非線形) — でゲートを低次元潜在 z ∈ R^n に圧縮する。
      + *Compress* the gates into a low-dim latent $z in RR^n$ with a #text(blue)[preprocessor] — PCA (linear) or an *autoencoder* (nonlinear).
      // 訳: 潜在の支配方程式を #text(red)[ansatz] (方程式の形の仮定) で同定する — 全項を候補ライブラリから選ぶ SINDy か、既知のイオン物理を残し潜在のみを SINDy で当てる hybrid。
      + *Identify* the latent's governing equation. The #text(red)[ansatz] sets the *assumed form*: *SINDy* picks terms from a generic candidate library, while *hybrid* keeps the known ionic physics and lets SINDy fit *only* the latent.
      $
        cases(
          frac(d V, d t) = underbrace(I_"ion"^"phys" (V, bold(z)), "known physics") + I_"ext",
          frac(d bold(z), d t) = underbrace(Theta(bold(z), V) bold(xi), "SINDy-identified"),
          bold(z) = "AE"_"enc" ("gates") \, quad "gates" = "AE"_"dec" (bold(z))
        )
      $
      // 訳: これは非物理項を多数含んだ昨年の過大 (41項) SINDy ライブラリを物理制約で直接修正するもの。
      #text(size: 21pt)[
        The #text(red)[hybrid] ansatz directly fixes last year's *over-large (41-term)* SINDy library, which contained many non-physical terms.
      ]

      #grid(
        columns: (1.3fr, 1fr),
        gutter: 0em,
        [
          #figure(
            image("pic/result/I_ext.png", width: 100%),
            caption: [Random 10 ms pulse currents used to drive the simulation],
            supplement: none,
            numbering: none,
          )<teaching-current>
        ],
        [
          #figure(
            image("pic/result/compress_traj_raw.png", width: 80%),
            caption: "Gate compression into latent space",
            supplement: none,
            numbering: none,
          )],
      )
    ],
    [
      // 訳: 対象: Traub CA3 錐体細胞
      *Target: Traub CA3 pyramidal cell*\
      // 訳: (19コンパートメントモデル — 昨年のトイ 3コンパートメントモデルに替わる現実的なマルチコンパートメントニューロン)
      (a 19-compartment model — a realistic multi-compartment neuron, replacing last year's toy 3-compartment model)

      #figure(
        image("pic/ref/traub_comp.png", width: 55%),
      )
      #v(0.3em)
      // 訳: soma コンパートメントをサロゲートで置換する。somaのイオンゲート変数を低次元潜在 z で表しつつ、コンパートメント間結合は物理のまま残す。
      *We replace the soma compartment with the surrogate*, so that the ionic gate variables of the soma are represented by the low-dimensional latent $bold(z)$ while the compartment coupling stays physical.
      #v(0.3em)
      #text(size: 22pt)[
        - preprocessor $in$ {PCA, AE}
        - ansatz $in$ {SINDy, hybrid}
        - latent dim $n$ = 2 (HH) / 3 (Traub)
      ]

    ],
    [
      #mini-box(title: "SINDy")[
        #figure(
          image("pic/ref/SINDy.png"),
        )
        $X$: time-series data\
        $Theta(X)$: library evaluated at $X$\

        // 訳: 係数行列 Ξ について $dot(X)=Theta(X)Xi$ をスパース回帰として解く。
        Solve $dot(X)=Theta(X)Xi$ as a sparse regression for the coefficient matrix $Xi$ @Champion-2019-DatadrivenDiscoveryCoordinatesGoverning
      ]
    ],
  )


]

// tighten figure spacing
#show figure: set block(spacing: 1em)
#show figure: set figure(gap: 0em)

#grid(
  columns: (3fr, 1fr),
  gutter: 1em,
  [

    #pop.column-box(heading: "Results and Discussion")[
      #set text(size: body-size)

      #grid(
        columns: (1fr, 1fr),
        gutter: 0.8em,
        // ---- Column ①: single HH proof of concept ----
        [
          *① Proof of concept — single HH soma*
          #grid(
            columns: (1.35fr, 1fr),
            gutter: 0.6em,
            [
              #figure(
                image("pic/result/hh_single/diff.png", width: 100%),
                caption: [HH surrogate (AE, $n=2$, hybrid): $V$, the two AE latents $g_1, g_2$, and the original 3 gates. Original (blue) vs. surrogate (red, dashed) overlap almost exactly.],
                numbering: none,
                supplement: none,
              )
            ],
            [
              #mini-box(title: "Waveform match", color: rgb("#2a7f2a"))[
                #set text(size: 21pt)
                - spike-shape corr = *1.000*
                - AP amplitude err = *0.5 mV*
                - RMSE = *1.0 mV*
              ]
              #v(0.3em)
              #figure(
                image("pic/result/hh_sweep/sweep.png", width: 100%),
                caption: [Spike count vs. input amplitude — matches the original over the whole range (not overfit to one stimulus).],
                numbering: none,
                supplement: none,
              )
            ],
          )
          #v(0.2em)
          // 訳: 3ゲート (m,h,n) → 2潜在: AE はゲートを2次元潜在に圧縮するが、サロゲートはスパイクをほぼ誤差ゼロで再現する。
          - *3 gates $(m,h,n) ->$ 2 latents*: the AE compresses the gates into a 2-D latent, yet the surrogate reproduces the spike with *near-zero error*.
          // 訳: hybrid ansatz はイオン物理を保持するため、同定された潜在ダイナミクスはコンパクトかつ安定に保たれる。
          - The hybrid ansatz retains the ionic physics, so the identified latent dynamics stay compact and stable.
        ],
        // ---- Column ②: scale-up to Traub 19-compartment ----
        [
          *② Scaling up — Traub 19-compartment (soma replaced)*
          #grid(
            columns: (1.15fr, 1fr),
            gutter: 0.6em,
            [
              #figure(
                image("pic/result/traub_single/diff.png", width: 100%),
                caption: [Traub19 soma replaced (AE, $n=3$, hybrid). $V$, the 3 latents, and the original soma gates. Original (blue) vs. surrogate (red, dashed).],
                numbering: none,
                supplement: none,
              )
            ],
            [
              #figure(
                image("pic/result/traub_sweep/sweep.png", width: 100%),
                caption: [Spike count across surrogate variants. Only *AE + $n=3$* tracks the original; PCA and $n=2$ collapse — motivating the nonlinear AE and the $n=3$ latent.],
                numbering: none,
                supplement: none,
              )
            ],
          )
          #v(0.2em)
          // 訳: 同じ AE + hybrid フレームワークが発散せず現実的なマルチコンパートメントモデルへ転移し、バースト発火し入力依存のスパイク数を追従する (spike-shape corr = 1.000)。
          - The *same AE + hybrid framework* transfers to a realistic multi-compartment model *without diverging*, firing in bursts and tracking the input-driven *spike count* (spike-shape corr = 1.000).
          // 訳: 定性的構造を捉える — 発火閾値とバースト開始が入力振幅に応じて変化し、原モデルと同様。
          - It captures the *qualitative structure* — firing threshold and burst onset shift with input amplitude, as in the original.
          // 訳: 残る課題: バースト後の減衰/終息はまだ再現できず、サロゲートは振動し続ける。
          - *Remaining gap*: the post-burst *decay/termination* is not yet reproduced — the surrogate keeps oscillating.
        ],
      )
    ]
  ],
  [

    #pop.column-box(heading: "Conclusion")[
      #set text(size: body-size)
      // 訳: 単一 HH ニューロンでは、AE + hybrid サロゲートが状態次元を半分以下 (4→2) にしつつスパイク波形をほぼ誤差ゼロで再現する。
      - For a single HH neuron, the *AE + hybrid* surrogate reproduces the spike waveform with *near-zero error* while halving+ the state dimension ($4 -> 2$).
      // 訳: 同じフレームワークが現実的な19コンパートメント Traub モデルへ発散せずスケールし、定量的にはまだだが定性的な発火挙動を捉える。
      - The same framework *scales to a realistic 19-compartment Traub model* without diverging and captures the qualitative firing behaviour, though not yet quantitatively.
      #v(1em)
      $->$ *Future work*\
      // 訳: マルチコンパートメントサロゲートのバースト後減衰ダイナミクスを再現し、物理インフォームドライブラリをさらに洗練する。
      Reproduce the post-burst decay dynamics of the multi-compartment surrogate, and further refine the physics-informed library.

    ]
    #set text(size: body-size)

    *Code link*\
    // 訳: 本研究のコードは以下で公開している。
    Code for this study is available at
    #link("https://github.com/MunechikaHaruki/SINDyNeuroSurrogate")
    #show bibliography: set text(size: 22.5pt)
    #bibliography("bibliography.bib")

  ],
)
