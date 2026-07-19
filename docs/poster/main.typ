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

#let body-size = 27pt

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
      - Multi-compartment models represent the spatial morphology of a neuron as connected compartments, reproducing neuronal activity in fine detail.
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
      $->$ When many multi-compartment models are simulated in parallel (e.g. brain-scale simulation), the gate variables cause a *memory bottleneck*.
      #v(0.3em)
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
      *Two-axis surrogate framework*
      + Simulate the target model; collect time series of membrane potential $V$ and gate variables.
      + *Compress* the gates into a low-dim latent $z in RR^n$ with a #text(blue)[preprocessor] — PCA (linear) or an *autoencoder* (nonlinear).
      + *Identify* the latent dynamics with an #text(red)[ansatz] — SINDy (full library) or *hybrid* (physics-separated).
      $
        cases(
          frac(d V, d t) = underbrace(I_"ion"^"phys" (V, bold(z)), "known physics") + I_"ext",
          frac(d bold(z), d t) = underbrace(Theta(bold(z), V) bold(xi), "SINDy-identified"),
          bold(z) = "AE"_"enc" ("gates") \, quad "gates" = "AE"_"dec" (bold(z))
        )
      $
      #text(size: 21pt)[
        The #text(red)[hybrid] ansatz keeps the *known ionic physics* and lets SINDy identify only the *latent* dynamics — a direct answer to last year's over-large (41-term) library that contained many non-physical terms.
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
      *Target: Traub CA3 pyramidal cell*\
      (a 19-compartment model — a realistic multi-compartment neuron, replacing last year's toy 3-compartment model)

      #figure(
        image("pic/ref/traub_comp.png", width: 55%),
      )
      #v(0.3em)
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
        columns: (1.15fr, 1fr),
        gutter: 0.6em,
        [
          *① Proof of concept — single HH neuron*
          #figure(
            image("pic/result/hh_hybrid_ae_n2.png", width: 100%),
            caption: [HH surrogate (AE, $n=2$, hybrid): membrane potential $V$, the two AE latents, and the original 3 gates. Original (blue) and surrogate (red) overlap almost exactly.],
            numbering: none,
            supplement: none,
          )
          #v(0.3em)
          - *3 gates $->$ 2 latents*: the AE compresses $(m,h,n)$ into a 2-D latent while the surrogate reproduces the spike with *near-zero error*.
          - The hybrid ansatz retains the ionic physics, so the identified latent dynamics stay compact and stable.
        ],
        [
          *② Scaling up — Traub 19-compartment*
          #figure(
            image("pic/result/traub_n3_excerpt.png", width: 100%),
            caption: [Traub19 soma replaced by the surrogate (AE, $n=3$, hybrid). Amplitude sweep of a steady input current; original (black) vs. surrogate (red).],
            numbering: none,
            supplement: none,
          )
          #v(0.3em)
          - The *same AE + hybrid framework* transfers to a realistic multi-compartment model *without diverging*.
          - It captures the *qualitative structure* — firing threshold and burst onset shift with input amplitude, as in the original.
          - *Remaining gap*: the post-burst *decay/termination* is not yet reproduced (the surrogate keeps oscillating).
        ],
      )
    ]
  ],
  [

    #pop.column-box(heading: "Conclusion")[
      #set text(size: body-size)
      - For a single HH neuron, the *AE + hybrid* surrogate reproduces the spike waveform with *near-zero error* while halving+ the state dimension ($4 -> 2$).
      - The same framework *scales to a realistic 19-compartment Traub model* without diverging and captures the qualitative firing behaviour, though not yet quantitatively.
      #v(1em)
      $->$ *Future work*\
      Reproduce the post-burst decay dynamics of the multi-compartment surrogate, and further refine the physics-informed library.

    ]
    #set text(size: body-size)

    *Code link*\
    Code for this study is available at
    #link("https://github.com/MunechikaHaruki/SINDyNeuroSurrogate")
    #show bibliography: set text(size: 22.5pt)
    #bibliography("bibliography.bib")

  ],
)
