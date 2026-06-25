


== 3Compartment代理モデルの作成方法
// #image("3comp.png", width: 100%)

somaコンパートメントはHodgkin-Huxleyモデルと同じように振る舞う

はじめにHodgkin-Huxley代理モデルを作り、これをsomaコンパートメントと置き換える

== Hodgkin-Huxley代理モデルの作成方法

+ Hodgkin-Huxleyモデルに電流を与えてシミュレーション(膜電位V(t)、隠れ変数のデータを得る)
+ 得られた隠れ変数のデータをPCAを用い１次元データ$g_0(t)$に圧縮
+ 時系列データ$V(t),g_0(t)$を再現する代理モデルをSINDyで同定

$
  cases(
    frac(d V, d t) = text(#red, a_11) ( text(#blue,alpha_m) (V) g_0) + text(#red, a_12) ( text(#blue,alpha_m) (V) I_("ext")) - text(#red, a_13) (text(#blue,alpha_m) (g_0) I_("ext")) + dots,
    frac(d g_0, d t) = text(#red, a_21) (text(#blue,alpha_m) (V) g_0) text(#red, a_22) (text(#blue,alpha_n) (V) I_("ext")) +text(#red, a_23)(text(#blue,alpha_m) (V)) + dots
  )
$
- SINDy(非線形ダイナミクスのスパース同定)で、元のダイナミクスを再現するよう上式の#text(red)[係数]推定
- 代理モデル内の#text(blue)[関数]はSINDyにおけるハイパーパラメーターで、Hodgkin-Huxleyモデル中に出現する関数群を選択



