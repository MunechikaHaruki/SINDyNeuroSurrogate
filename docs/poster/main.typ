#import "@preview/peace-of-posters:0.5.6" as pop
#import "@preview/typsium:0.3.1": *

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
    columns: (1fr, 0.82fr, 0.72fr),
    gutter: 0.5em,
    [
      // 訳: 2軸サロゲートフレームワーク
      *Two-axis surrogate framework*
      // 訳: 対象をシミュレーションし V とゲートの時系列を集める。
      + Simulate the target; collect $V$ + gate time series.
      // 訳: preprocessor (PCA / autoencoder) でゲートを低次元潜在 z へ圧縮。
      + *Compress* gates into a low-dim latent $z in RR^n$ via a #text(blue)[preprocessor] — PCA (linear) or an *autoencoder* (nonlinear).
      // 訳: 潜在の支配式を ansatz で同定。SINDy=汎用ライブラリから項選択、hybrid=イオン物理を残し潜在のみ当てる。
      + *Identify* the latent dynamics. The #text(red)[ansatz] sets the form: *SINDy* picks from a generic library; *hybrid* keeps the ionic physics, fitting *only* the latent.
      $
        cases(
          frac(d V, d t) = underbrace(I_"ion"^"phys" (V, bold(z)), "known physics") + I_"ext",
          frac(d bold(z), d t) = underbrace(Theta(bold(z), V) bold(xi), "SINDy-identified"),
          bold(z) = "AE"_"enc" ("gates") \, quad "gates" = "AE"_"dec" (bold(z))
        )
      $
      // 訳: hybrid は昨年の過大 (41項) SINDy ライブラリを直接修正。
      #text(size: 22pt)[
        #text(red)[hybrid] fixes last year's *over-large (41-term)* SINDy library.
      ]
    ],
    [
      // 訳: 対象: Traub CA3 錐体細胞 — 19コンパートメントの現実的ニューロン (昨年のトイ 3-comp に替わる)。
      *Target: Traub CA3 pyramidal cell* — a 19-compartment realistic neuron (vs. last year's toy 3-comp).

      #figure(
        image("pic/ref/traub_comp.png", width: 60%),
      )
      #v(0.2em)
      // 訳: soma をサロゲートで置換 — soma ゲート→潜在 z、コンパートメント結合は物理のまま。
      *Replace the soma with the surrogate*: soma gates $-> bold(z)$; compartment coupling stays physical.
      #v(0.2em)
      #text(size: 22pt)[
        - preprocessor $in$ {PCA, AE}
        - ansatz $in$ {SINDy, hybrid}
        - latent dim $n$ = 2 (HH) / 3 (Traub)
      ]
    ],
    [
      #mini-box(title: "SINDy")[
        #figure(image("pic/ref/SINDy.png"))
        // 訳: 係数行列 Ξ を $dot(X)=Theta(X)Xi$ のスパース回帰で解く。
        Solve $dot(X)=Theta(X)Xi$ (sparse regression) for $Xi$ @Champion-2019-DatadrivenDiscoveryCoordinatesGoverning
      ]
    ],
  )
]

// tighten figure spacing
#show figure: set block(spacing: 1em)
#show figure: set figure(gap: 0em)

#pop.column-box(heading: "Results and Discussion")[
  #set text(size: 24pt)
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 1em,
    // ======== ① MODEL SELECTION : amp sweep -> freq sweep -> hybrid/n2/ae ========
    [
      *① Model selection — why hybrid/n2/ae?*
      // 訳: 変数選択を2つの sweep で絞り込む。まず amp sweep で潜在次元、次に freq sweep で preprocessor。
      #v(0.2em)
      #figure(
        image("pic/result/HH/sweep_traces.png", width: 60%),
        caption: [Amplitude sweep across variants. *$n=2$* (ae, pca) reproduces the spike; *$n=1$* collapses.],
        numbering: none,
        supplement: none,
      )
      // 訳: 潜在次元 → n=2 が必須。n=1 は発火できない。
      - *Latent dim*: $n=2$ is required — $n=1$ cannot fire $-> $ drop $n=1$.
      #v(0.2em)
      #figure(
        image("pic/result/HH_sinousoidal/sweep.png", width: 85%),
        caption: [Frequency sweep (*sinusoidal* drive, unseen). *ae* tracks the original; *pca* mis-fires at $f=0$ and dips near 110 Hz.],
        numbering: none,
        supplement: none,
      )
      // 訳: preprocessor → AE。非線形エンコーダが 0–200 Hz を追従、PCA は失敗。
      - *Preprocessor*: AE over PCA — the nonlinear encoder tracks 0–200 Hz $-> $ drop pca.
      #v(0.2em)
      #mini-box(title: "Selected", color: rgb("#2a7f2a"))[
        #set text(size: 24pt)
        // 訳: 2つの sweep から hybrid / n=2 / AE を選択。
        $-> $ *hybrid / $n=2$ / AE*
      ]
    ],
    // ======== ② SINGLE HH : accuracy + interpretability ========
    [
      *② Single HH — accurate & interpretable*
      #figure(
        image("pic/result/HH/diff.png", width: 74%),
        caption: [hybrid/n2/ae under a 10 ms pulse: $V$, the 2 AE latents $g_1, g_2$, and the 3 original gates. Original (blue) vs. surrogate (red, dashed) overlap.],
        numbering: none,
        supplement: none,
      )
      #mini-box(title: "Waveform match", color: rgb("#2a7f2a"))[
        #set text(size: 21pt)
        - RMSE *1.0 mV*, MAE *0.4 mV*, AP amp err *0.5 mV*
        - spike count & latency *exact*
      ]
      // 訳: 3ゲート → 2潜在で、スパイクをほぼ誤差ゼロで再現。
      - *3 gates $-> $ 2 latents*, spike reproduced with *near-zero error*.
      #v(0.2em)
      // 訳: hybrid はスパースで解釈可能な潜在方程式を同定 (HH レート関数の少数項)。
      #figure(
        image("pic/result/HH/model.png", width: 100%),
        caption: [*Sparse, interpretable* identified model: each $dot(g)_i$ is a few HH-rate-function terms.],
        numbering: none,
        supplement: none,
      )
    ],
    // ======== ③ SCALE-UP : Traub + Conclusion + Code ========
    [
      *③ Scaling up — Traub 19-compartment*
      #figure(
        image("pic/result/Traub/sweep_traces.png", width: 82%),
        caption: [Spike count vs. amplitude, surrogate variants (soma replaced).],
        numbering: none,
        supplement: none,
      )
      // 訳: 同じ枠組みが19コンパートメントへ発散せず転移。AE+n=3 のみ発火、PCA/n=2 は潰れる。数は過大 (9 vs 6)。
      - Same framework transfers *without diverging*; *only AE + $n=3$* fires (PCA, $n=2$ collapse).
      - *Gap*: spike count *overestimated* (9 vs. 6) — qualitative, not yet quantitative.

      #v(0.3em)
      #mini-box(title: "Conclusion")[
        #set text(size: 24pt)
        // 訳: HH: 状態 4→2 でほぼ誤差ゼロ、正弦波へ汎化、スパースな潜在式を同定。
        - HH: near-zero error at *dim $4 -> 2$*, generalizes to sinusoid, *sparse* latent eq.
        // 訳: 現実的 Traub 19-comp へ発散せずスケール、定性的発火を再現 (定量化は今後)。
        - Scales to *Traub 19-comp* without diverging; qualitative firing recovered.
        // 訳: 今後: スパイク数ギャップを詰め、物理インフォームドライブラリを洗練。
        $->$ *Future*: close the spike-count gap, refine the physics-informed library.
      ]

      #v(0.3em)
      #text(size: 22pt)[
        *Code* — #link("https://github.com/MunechikaHaruki/SINDyNeuroSurrogate")
      ]
      #show bibliography: set text(size: 19pt)
      #bibliography("bibliography.bib", title: none)
    ],
  )
]
