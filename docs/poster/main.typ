#import "@preview/peace-of-posters:0.5.6" as pop
#import "@preview/typsium:0.3.1": *

#set page("a0", margin: 2cm)
#pop.set-poster-layout(pop.layout-a0)
#pop.set-theme(pop.uni-fr)
#set text(size: pop.layout-a0.at("body-size"))
#set text(font: ("Hiragino Kaku Gothic ProN", "New Computer Modern"))
#let box-spacing = 1em
#set columns(gutter: box-spacing)
#set block(spacing: box-spacing)
#pop.update-poster-layout(spacing: box-spacing)

#set text(lang: "ja")

#let body-size = 27pt

// --- ここから：小箱を作る関数の定義 ---
#let mini-box(title: "小見出し", color: blue, body) = {
  block(
    width: 100%,
    stroke: 1pt + color, // 枠線の色
    radius: 4pt, // 角の丸み
    clip: true, // 丸みからはみ出さないようにする
    stack(
      dir: ttb, // 上から下に積む
      // ヘッダー部分
      block(
        width: 100%,
        fill: color,
        inset: 6pt,
        text(fill: white, weight: "bold", size: 35pt)[#title],
      ),
      // コンテンツ部分
      block(
        width: 100%,
        fill: white, // 背景を白にする（必要ならluma(250)などで薄いグレーに）
        inset: 8pt,
        body,
      ),
    ),
  )
}



#pop.title-box(
  text(size: 60pt)[
    ニューロンシミュレーションの計算コスト削減を目指したサロゲートモデルの開発 #v(-3em)
  ],
  authors: "Haruki Munechika",
  institutes: "",
)



#pop.column-box(heading: "Introduction")[
  #set text(size: body-size)
  #grid(
    columns: (1.5fr, 1fr),
    // 1:1 の比率で分割 (3列なら 1fr, 1fr, 1fr)
    gutter: 1em,
    // 列間の隙間
    [
      - ニューロンの空間形状を連結するコンパートメントとして表現したMulti-Compartmentモデルであれば、ニューロンの活動の詳細な再現が可能
      - 各コンパートメントはHodgkin-Huxleyモデルのようにゲート変数(イオンチャネルの開口率)を持つ微分方程式で記述
        #figure(
          // ここに数式を書く
          $
            C_m frac(d V, d t) = -overline(g)_L (V - E_L) - overline(g)_#ce("Na") m^3h (V - E_#ce("Na") ) - overline(g)_#ce("K") n^4 (V - E_#ce("K")) + I_"ext" \
          $,

          // 説明文
          caption: [Hodgkin-Huxleyモデルの膜電位変化式(m,h,n:ゲート変数)],

          // 重要: これで「図」ではなく「式」として扱われる
          kind: "equation",
          numbering: none,
          supplement: none,
        ) <eq-hh> // ラベル
      $->$ 脳シミュレーションのようにMulti-Compartmentモデルを並列に計算する場合、ゲート変数の存在によるメモリ不足が問題となる
      #v(0.3em)
      #mini-box(title: "目的")[
        Multi-Compartmentニューロンモデルの膜電位応答を再現ができるサロゲートモデルの作成
      ]

    ],
    [
      #grid(
        columns: (1fr, 1fr),
        gutter: 1em,
        [
          #figure(
            image("pic/pyramidal_.png", width: 60%),
            caption: [錐体ニューロン @noauthor__2012],
            numbering: none,
            supplement: none
          )
          #figure(
            image("pic/traub_comp.png", width: 60%),
            caption: none,
            numbering: none,
            supplement: none
          )<comp>

          #figure(
            image("pic/traub_withcap.png",width: 70%),
            caption: [Multi-Compartmentモデルの空間表現とシミュレーション@Traub-1991-ModelCA3HippocampalPyramidal],
            numbering: none,
            supplement: none
          )<diff>
        ],
        [
          #figure(
            image("pic/yamazaki.png"),
            caption: [Hodgkin-Huxleyモデル膜電位応答@山﨑匡-2021-はじめて],
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
      *Hodgkin-Huxley代理モデル*
      + Hodgkin-Huxleyモデルのシミュレーションをおこない、膜電位$V$、ゲート変数$m,h,n$の時系列データを取得
      + 得られたゲート変数のデータをPCAで１次元データ$g_0(t)$に圧縮
      + 時系列データ$V(t),g_0(t)$を再現する代理モデルを同定(SINDy)
      $
        cases(
          frac(d V, d t) = text(#red, a_11) ( text(#blue, alpha_m) (V) g_0) + text(#red, a_12) ( text(#blue, alpha_m) (V) I_("ext")) - text(#red, a_13) (text(#blue, alpha_m) (g_0) I_("ext")) + dots,
          frac(d g_0, d t) = text(#red, a_21) (text(#blue, alpha_m) (V) g_0) text(#red, a_22) (text(#blue, alpha_n) (V) I_("ext")) +text(#red, a_23)(text(#blue, alpha_m) (V)) + dots
        )
      $
      #text(size: 21pt)[
        #text(blue)[関数]はハイパーパラメーターとしてHodgkin-Huxleyを構成する関数(e.g.,$text(#blue, alpha_h (x)=exp(-V/20))$)をメインに選び、#text(red)[係数推定]をダイナミクスを再現できるよう実施
      ]



      #grid(
        columns: (1.3fr, 1fr),
        gutter: 0em,
        [


          #figure(
            image("pic/I_ext.png", width: 100%),
            caption: [シミュレーション時に与えた10ms幅のランダムなパルス電流],
            supplement: none,
            numbering: none,
          )<teaching-current>

        ],
        [
          #figure(
            image("pic/compress_traj_raw.png", width: 80%),
            caption: "PCAによる圧縮の様子",
            supplement: none,
            numbering: none,
          )],
      )
    ],
    [


      *3Compartmentモデル*\
      (Hodgkin-Huxleyモデルを拡張した単純なMulti-Compartmentモデル)

      #figure(
        image("pic/3comp.png"),
      )
      $
        cases(
          C frac(d V_"dend", d t)=-g_"leak"(V-E_"rest") && -I_"pre" && +I_"ext",
          C frac(d V_"soma", d t)=-g_"leak"(V-E_"rest") & + text(#blue, I_"ion-HodgkinHuxley")(V_"soma","gates") & + I_"pre"&-I_"post" &,
          C frac(d V_"axon", d t)=-g_"leak"(V-E_"rest") &&& +I_"post" &
        )
      $
      *somaコンパートメントをHodgkin-Huxley代理モデルに置き換えることで、3Compartmentモデルの代理モデルとする*

    ],
    [
      #mini-box(title: "SINDy")[
        #figure(
          image("pic/SINDy.png"),
        )
        $X$:時系列データ\
        $Theta(X)$:基底関数にXを代入\

        $dot(X)=Theta(X)Xi$の回帰問題として係数行列$Xi$を求める@Champion-2019-DatadrivenDiscoveryCoordinatesGoverning
      ]
    ],
  )


]

// 図や式の上下の余白を詰める設定
#show figure: set block(spacing: 1em)

// キャプションと図の距離を詰める設定
#show figure: set figure(gap: 0em)

#grid(
  columns: (3fr, 1fr),
  gutter: 1em,
  [

    #pop.column-box(heading: "Results and Discussion")[
      #set text(size: body-size)

      #grid(
        columns: (2fr, 1fr),
        gutter: 0em,
        rows: 50.7cm,
        inset: (left: -3em),
        [
          #figure(
            image("pic/compress.png"),
            caption: "シミュレーションの結果(左)と教師データ(右)",
            numbering: none,
            supplement: none,
          )

          #figure(
            $
              cases(
                frac(d V, d t) = text(#red, 7.28) ( text(#blue, alpha_m) (V) g_0) + text(#red, 0.01) ( text(#blue, alpha_m) (V) I_("ext")) - text(#red, 4.19) (text(#blue, alpha_m) (g_0) I_("ext")) + dots,
                frac(d g_0, d t) = text(#red, -0.02) (text(#blue, alpha_m) (V) g_0) text(#red, - 0.06) (text(#blue, alpha_n) (V) I_("ext")) +text(#red, 3.91)(text(#blue, alpha_m) (V)) + dots
              )
            $,
            caption: [Hodgkin-Huxley代理モデル(変数の数:$4->2$)],
            kind: "equation",
            numbering: none,
            supplement: none,
          )

          // $4->2$


          #figure(
            image("pic/teaching.png"),
          )
          #figure(
            image("pic/long_rand.png"),
          )
          #h(4em) - 変数の数が$(4 ->2)$で半減しているので、必要なメモリの量は半減\
          #h(4em) - SINDyにより、大まかなダイナミクスは再現可能

        ],
        [
          #figure(
            image("pic/three_comp_result.png", width: 100%),
            caption: [3Compartmentモデルにパルス電流を加えた結果],
            numbering: none,
            supplement: none,
          )
          #figure(
            image("pic/steady_hh3_20_processed.png"),
            caption: none,
            numbering: none,
            supplement: none,
          )
          #v(-0.5em)
          #h(-1.5em)- somaコンパートメントに流入する電流は隣接するコンパートメントとの電位差により決定\
          #h(-1.5em)- スパイク発射の閾値などを正確に再現する必要があるが、Multi-Compartmentモデルに組み込むには精度不足
        ],
      )
    ]
  ],
  [

    #pop.column-box(heading: "Conclusion")[
      #set text(size: body-size)
      - SINDyによってニューロンのダイナミクス同定が可能であることが示せた
      - Multi-Compartmentモデルに代理モデルを組み混んだ場合、定常電流を加えた場合の周期性は再現できた。しかし、実用的なものにするにはより精度が必要
      #v(1em)
    $->$基底関数の見直し\
        現在の基底関数は41個もあり、 $alpha_m (g_0) I_("ext")$などの、元のHodgkin-Huxleyモデルにはない項も多く含んでいるため

    ]
    #set text(size: body-size)
 
    *Code link*\
    本研究に用いたコードは以下に公開します
    #link("https://github.com/MunechikaHaruki/SINDyNeuroSurrogate")
    #show bibliography: set text(size: 22.5pt)
    #bibliography("bibliography.bib")

  ],
)
