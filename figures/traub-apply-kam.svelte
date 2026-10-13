<script lang="ts">
  import { eq, tex } from '$lib/tex.ts';
  import Card from '$lib/Card.svelte';
  import LineChart from '$lib/LineChart.svelte';
  import Page from '$lib/Page.svelte';
  import Toggles from '$lib/Toggles.svelte';
  import { equilibria, flatRanges, gateNames, groupCriteria, rates, sensitivity, speed, steady, steadyCurrent, steadySlope, voltageCriteria, type GateName } from './gate-reduction-model.ts';

  type Point = { x: number; y: number };
  type Trace = { label: string; points: Point[] };
  const voltageLabel = '膜電位 V (mV)';

  // ---- 平衡点：定常 I-V 曲線 ----
  // 端の −75, 55 mV では感度が 0 になり重みが定義できないので、内側だけを描く。
  const voltageDomain: [number, number] = [-75, 55];
  const grid: number[] = [];
  for (let i = 0; i < 259; i++) grid.push(-74.5 + i * 0.5);
  /** I₀ = 0 の平衡点。式(10)を読むのは、このうち系が留まる静止電位（左端）。 */
  const rest = equilibria(0);
  const steadyCurve: Point[] = [];
  for (const v of grid) steadyCurve.push({ x: v, y: steadyCurrent(v) });
  const restLines = { label: '平衡点', tone: 'accent' as const, x: rest };
  const currentLine = { label: 'I₀ = 0', tone: 'accent' as const, y: [0] };

  // ---- 条件 1：ゲートの定常値 ----
  const gateDomain: [number, number] = [-100, 60];
  /** 8 つのゲートの定常値曲線 x̄(V)（等価電位は、これを逆に読む）。 */
  function steadyTraces(): Trace[] {
    const list: Trace[] = [];
    for (const name of gateNames) {
      const points: Point[] = [];
      for (let v = gateDomain[0]; v <= gateDomain[1]; v += 0.5) points.push({ x: v, y: steady(rates(v)[name]) });
      list.push({ label: name, points });
    }
    return list;
  }
  const steadyCurves = steadyTraces();
  /** 8 つのゲートの定常値の傾き x̄'(V)。変換できる条件 x̄' ≠ 0 を見る。 */
  function slopeTraces(): Trace[] {
    const list: Trace[] = [];
    for (const name of gateNames) {
      const points: Point[] = [];
      for (let v = gateDomain[0]; v <= gateDomain[1]; v += 0.5) points.push({ x: v, y: steadySlope(name, v) });
      list.push({ label: name, points });
    }
    return list;
  }
  const slopeCurves = slopeTraces();
  /** x̄ が平らな範囲を「r：−100 〜 −60 mV」の形に並べる。平らな範囲のないゲートは省く。 */
  function flatSummary(): string {
    const parts: string[] = [];
    for (const name of gateNames) {
      for (const range of flatRanges(name, gateDomain[0], gateDomain[1], 0.1)) {
        parts.push(`${name}：${range.low} 〜 ${range.high} mV`);
      }
    }
    return parts.join('、') || 'なし';
  }
  const flatText = flatSummary();

  // ---- 式(10)・(11)：判定 ----
  /** 式(10)で組を選ぶ、感度が正の ψ 群の候補。 */
  const recoveryGates: GateName[] = ['h', 'n', 'a', 'r', 'c'];
  /** 式(11)で一つずつ見る、感度が負の φ 群の候補。式(11)はゲートごとの式なので、組は選ばない。 */
  const voltageGates: GateName[] = ['m', 's', 'b'];
  /** ゲートの並びを Toggles の選択肢に直す。 */
  function optionsOf(names: GateName[]): { value: GateName; label: string }[] {
    const options: { value: GateName; label: string }[] = [];
    for (const name of names) options.push({ value: name, label: name });
    return options;
  }
  /** 選んだゲートを候補の並びにそろえる。色が選んだ順で入れ替わらないようにする。 */
  function ordered(selection: GateName[], order: GateName[]): GateName[] {
    const names: GateName[] = [];
    for (const name of order) if (selection.includes(name)) names.push(name);
    return names;
  }
  let recoveryShown = $state<GateName[]>(['h', 'n']);
  let recovery = $derived(ordered(recoveryShown, recoveryGates));
  /** 式(11)の左辺を、ゲートごとに 1 本。 */
  function ratioTraces(names: GateName[]): Trace[] {
    const list: Trace[] = [];
    for (const name of names) {
      const points: Point[] = [];
      for (const v of grid) points.push({ x: v, y: voltageCriteria(name, v).ratio });
      list.push({ label: `式(11)・${name}`, points });
    }
    return list;
  }
  /** 静止電位での式(10)の左辺と式(6)の重み。式(10)は平衡点で導いた式なので、静止電位でだけ読む。 */
  let atRest = $derived(recovery.length >= 2 ? groupCriteria(recovery, rest[0]) : undefined);
  /** 群の判定量の最大値。 */
  function largest(values: number[]): number {
    let value = 0;
    for (const v of values) value = Math.max(value, v);
    return value;
  }
  const ratios = ratioTraces(voltageGates);
  const unitLine = { label: '1', tone: 'alert' as const, y: [1] };
  const zeroLine = { label: '0', tone: 'muted' as const, y: [0] };

  // ---- ゲートの速さ k ----
  /** 8 つのゲートの速さ k_x(V) = α_x + β_x（ms⁻¹）。 */
  function speedTraces(): Trace[] {
    const list: Trace[] = [];
    for (const name of gateNames) {
      const points: Point[] = [];
      for (let v = gateDomain[0]; v <= gateDomain[1]; v += 0.5) points.push({ x: v, y: speed(rates(v)[name]) });
      list.push({ label: name, points });
    }
    return list;
  }
  const speedCurves = speedTraces();

  // ---- 感度 D の値 ----
  /** 感度を描くゲート。並びは感度の符号の表と同じ。 */
  const sensitivityGates: GateName[] = ['m', 'h', 'n', 'a', 'b', 's', 'r', 'c'];
  /** 全状態が V で定常な点での D_x = (∂I/∂x) x̄'(V) を、ゲートごとに 1 本。 */
  function sensitivityCurves(names: GateName[]): Trace[] {
    const list: Trace[] = [];
    for (const name of names) {
      const points: Point[] = [];
      for (const v of grid) points.push({ x: v, y: sensitivity(name, v) });
      list.push({ label: `D・${name}`, points });
    }
    return list;
  }
  const sensitivityValues = sensitivityCurves(sensitivityGates);
</script>

<Page title="Traub 単一コンパートメントへの Kepler–Abbott–Marder 法の適用" summary="Traub の単一区画の 8 つのゲートを、Kepler–Abbott–Marder 法で群に分けて縮約する。等価電位と感度の符号で群を作れるゲートを絞り、近似のよさ（Kepler 式(10)・(11)）を膜電位の関数として判定し、縮約モデルで各時刻に計算する式を書く。" complete={true}>
  <p><small>典拠：<a href="https://doi.org/10.1007/BF00197717">Kepler et al. (1992)</a>、<a href="https://doi.org/10.1152/jn.1991.66.2.635">Traub et al. (1991)</a>。前提：<a href="../../../resources/neuron-models/kepler-abbott-marder">別冊「Kepler–Abbott–Marder のモデル縮約」</a>、<a href="../../../resources/neuron-models/traub-model">別冊「Traub の CA3 ニューロンモデル」</a>。</small></p>

  <h2>Traub の膜と 8 つのゲート</h2>
  <p>Traub の局所の膜方程式を細胞体のパラメータで使い、外部電流だけで駆動する一つの膜を扱う。状態は膜電位 {@html tex`V`}、電位で動く 8 つのゲート、Ca 系 {@html tex`\chi,q`} の 11 個である。</p>

  <h3>膜電位と七つの電流</h3>
  <p>外向き電流を正とし、電流は µA/cm²、{@html tex`C=3`} µF/cm² の密度で書く。</p>
  {@html eq`C\dot V = I_{\text{ext}}(t) - I_{\text{ion}}, \qquad I_{\text{ion}} = I_{\text L}+I_{\text{Na}}+I_{\text{Ca}}+I_{\text{DR}}+I_{\text A}+I_{\text{AHP}}+I_{\text C}`}
  <p>各電流は「最大コンダクタンス × ゲート × 電位差」の形をしている（Traub 式(2)）。{@html tex`g`} は mS/cm²、逆転電位は mV である。</p>
  <table>
    <thead><tr><th>電流</th><th>式</th><th>{@html tex`g`}</th><th>逆転電位</th></tr></thead>
    <tbody>
      <tr><td>{@html tex`I_{\text L}`} リーク</td><td>{@html tex`g_{\text L}(V-E_{\text L})`}</td><td>0.1</td><td>{@html tex`E_{\text L}=-60`}</td></tr>
      <tr><td>{@html tex`I_{\text{Na}}`} Na</td><td>{@html tex`g_{\text{Na}}m^2h(V-E_{\text{Na}})`}</td><td>30</td><td>{@html tex`E_{\text{Na}}=55`}</td></tr>
      <tr><td>{@html tex`I_{\text{Ca}}`} Ca</td><td>{@html tex`g_{\text{Ca}}s^2r(V-E_{\text{Ca}})`}</td><td>4</td><td>{@html tex`E_{\text{Ca}}=80`}</td></tr>
      <tr><td>{@html tex`I_{\text{DR}}`} 遅延整流 K</td><td>{@html tex`g_{\text{DR}}n(V-E_{\text K})`}</td><td>15</td><td>{@html tex`E_{\text K}=-75`}</td></tr>
      <tr><td>{@html tex`I_{\text A}`} A 型 K</td><td>{@html tex`g_{\text A}ab(V-E_{\text K})`}</td><td>5</td><td>{@html tex`E_{\text K}=-75`}</td></tr>
      <tr><td>{@html tex`I_{\text{AHP}}`} AHP 型 K</td><td>{@html tex`g_{\text{AHP}}q(V-E_{\text K})`}</td><td>0.8</td><td>{@html tex`E_{\text K}=-75`}</td></tr>
      <tr><td>{@html tex`I_{\text C}`} C 型 K</td><td>{@html tex`g_{\text C}c\min(1,\chi/250)(V-E_{\text K})`}</td><td>10</td><td>{@html tex`E_{\text K}=-75`}</td></tr>
    </tbody>
  </table>

  <h3>ゲートの速さと定常値</h3>
  <p>電位で動くゲート {@html tex`x\in\lbrace m,h,n,a,b,s,r,c\rbrace`} は、開くレート {@html tex`\alpha_x(V)`} と閉じるレート {@html tex`\beta_x(V)`}（Traub 式(4)）から、速さ {@html tex`k_x`} で定常値 {@html tex`\bar x`} へ近づく。</p>
  {@html eq`\dot x = k_x(V)\,[\bar x(V)-x], \qquad k_x=\alpha_x+\beta_x, \qquad \bar x=\frac{\alpha_x}{\alpha_x+\beta_x}`}
  <Card title="ゲートの速さ k" subtitle="k_x(V) = α_x + β_x。縦軸は対数">
    <LineChart traces={speedCurves} xDomain={gateDomain} xLabel={voltageLabel} yLabel="k (ms⁻¹)" yScale="log" />
    {#snippet info()}
      <p>{@html tex`1/k_x`} がゲートの時定数（ms）である。レートは Traub 式(4)をそのまま使う。</p>
    {/snippet}
  </Card>
  <Card title="ゲートの定常値 x̄" subtitle="x̄_x(V) = α_x / (α_x + β_x)">
    <LineChart traces={steadyCurves} xDomain={gateDomain} yDomain={[0, 1]} xLabel={voltageLabel} yLabel="定常値 x̄" />
    {#snippet info()}
      <p>{@html tex`r`} と {@html tex`c`} のレートは電位の範囲で式が切り替わり、片側で {@html tex`\beta=0`} となる（{@html tex`u=V+60`}）。</p>
      {@html eq`\alpha_r=\begin{cases}0.005 & u\le0\\ 0.005e^{-u/20} & u>0\end{cases}`}
      {@html eq`\beta_r=\begin{cases}0 & u\le0\\ 0.005-\alpha_r & u>0\end{cases}`}
      {@html eq`\bar{r}=\begin{cases}1 & u\le0\\ e^{-u/20} & u>0\end{cases}`}
      {@html eq`\alpha_c=\begin{cases}e^{(u-10)/11-(u-6.5)/27}/18.975 & u\le50\\ 2e^{-(u-6.5)/27} & u>50\end{cases}`}
      {@html eq`\beta_c=\begin{cases}2e^{-(u-6.5)/27}-\alpha_c & u\le50\\ 0 & u>50\end{cases}`}
      {@html eq`\bar{c}=\begin{cases}e^{(u-10)/11}/37.95 & u\le50\\ 1 & u>50\end{cases}`}
      <p>{@html tex`\beta=0`} の側では {@html tex`\bar{x}=\alpha/(\alpha+0)=1`} が厳密に成り立ち、傾きが 0 になる。残りの 6 つのレートは全電位で一つの式で、定常値は 0 や 1 に漸近するだけなので、平らな範囲を持たない。</p>
      <p>平らな範囲は、0.1 mV の格子で隣の点と {@html tex`\bar{x}`} が等しい区間をつないで拾う。{@html tex`\bar{c}`} は −10 mV の直下で 1 をわずかに超えるので、区間は −9.9 mV から始まる。</p>
    {/snippet}
  </Card>

  <h3>Ca 系</h3>
  <p>Ca 系 {@html tex`\chi,q`} は電位で動くゲートではなく、Ca 電流で動く（Traub 式(3)）。</p>
  {@html eq`\dot\chi=-\kappa I_{\text{Ca}}-0.075\chi, \qquad \dot q=\min(0.00002\chi,\,0.01)(1-q)-0.001q`}

  <h2>等価電位</h2>
  <p>各ゲート {@html tex`x`} を、それを定常値に持つ電位 {@html tex`v_x`} で表す（Kepler 規則 1）。ゲートの式 {@html tex`\dot x=k_x(V)[\bar x(V)-x]`} に {@html tex`x=\bar x(v_x)`} を入れると、</p>
  {@html eq`\dot v_x = f_x(V,v_x) = \frac{k_x(V)\,[\bar x(V)-\bar x(v_x)]}{\bar x'(v_x)}`}
  <p>となる。</p>

  <h3>変換できる条件</h3>
  {@html eq`\bar x'(v_x)\neq0`}
  <p>{@html tex`\bar x`} が平らな電位では {@html tex`v_x`} が一つに決まらず、{@html tex`f_x`} の分母も 0 になる。</p>

  <p>8 つのゲートの定常値の傾き {@html tex`\bar x'(V)`} を描き、0 になる範囲を拾う。</p>
  <Card title="定常値の傾き x̄′" subtitle={`x̄′ = 0 となる範囲：${flatText}`}>
    <LineChart traces={slopeCurves} xDomain={gateDomain} xLabel={voltageLabel} yLabel="x̄′ (mV⁻¹)" guides={[zeroLine]} />
    {#snippet info()}
      <p>{@html tex`c`} の傾きは、場合分けの式をそのまま微分して求める（{@html tex`u=V+60`}）。</p>
      {@html eq`\bar c'=\begin{cases}\bar c/11 & u\le50\\ 0 & u>50\end{cases}`}
      <p>{@html tex`\bar c`} は境界の −10 mV で 1.000094 となり、その先の 1 へわずかに跳ぶ（Traub の定数 37.95 の丸めによる）。差分でこの跳びをまたぐと偽の負の傾きが出るので、{@html tex`c`} だけは解析的に求める。残りの 7 つは幅 ±10⁻⁴ mV の中心差分で求める。{@html tex`r`} も −60 mV で式が切り替わるが、{@html tex`\bar r`} は境界でつながっているので、差分は 0 と傾きの間の値を出すだけである。</p>
      <p>平らな範囲は、0.1 mV の格子で隣の点と {@html tex`\bar{x}`} が等しい区間をつないで拾う。{@html tex`\bar c`} は −10 mV で 1 をわずかに超えるので、区間は −9.9 mV から始まる。</p>
    {/snippet}
  </Card>
  <p>{@html tex`\bar r`} は {@html tex`v\le-60`}、{@html tex`\bar c`} は {@html tex`v>-10`} で平らになり、そこでは等価電位に変換できない。残りの 6 つは全電位で変換できる。</p>

  <h2>群分け</h2>
  <p>群分けは、ゲートの集合 {@html tex`G=\lbrace m,h,n,a,b,s,r,c\rbrace`} を次のように分ける選び方である。</p>
  {@html eq`G=\Phi\cup\Psi_1\cup\cdots\cup\Psi_K`}
  <ul>
    <li><b>φ 群 {@html tex`\Phi`}。</b>膜電位と一緒に、代表電位 {@html tex`\phi=\alpha_0V+\sum_{\mu\in\Phi}\alpha_\mu v_\mu`} にまとめる（Kepler 規則 2）。{@html tex`\phi`} が縮約後の膜電位になる。</li>
    <li><b>ψ 群 {@html tex`\Psi_k`}。</b>ゲート同士で、回復変数 {@html tex`\psi_k=\sum_{i\in\Psi_k}\alpha_iv_i`} にまとめる（Kepler 規則 3）。</li>
    <li><b>1 つだけの ψ 群 {@html tex`\lbrace x\rbrace`}。</b>{@html tex`\alpha_x=1`}、{@html tex`\psi=v_x`} で、元のゲートをそのまま残す。</li>
  </ul>
  <p>Kepler の方法は、定常値が Ca 濃度にも依存するゲートを対象にしていない（p.382）。そこで {@html tex`\chi,q`} は群分けに入れず、そのまま残す。縮約後の状態は</p>
  {@html eq`z=(\phi,\psi_1,\dots,\psi_K,\chi,q), \qquad \text{状態の数}=1+K+2`}
  <p>で、ψ 群の数 {@html tex`K`} で決まる。{@html tex`\Phi`} が空で 8 つのゲートを一つずつの群にすると {@html tex`K=8`} で、元の 11 状態に戻る。本研究の目標は、{@html tex`m,h,n,a,b,c`} の 6 つを 3 つの変数にまとめる群分けである。</p>

  <h2>感度</h2>
  <p>膜電流を等価電位で書いた {@html tex`F(V,\lbrace v_x\rbrace):=I_{\text{ion}}`} を、膜電位と各等価電位で微分したものを感度と呼ぶ。</p>
  {@html eq`D_0=\frac{\partial F}{\partial V}=g_{\text L}+g_{\text{Na}}m^2h+g_{\text{Ca}}s^2r+g_{\text{DR}}n+g_{\text A}ab+g_{\text{AHP}}q+g_{\text C}c\min(1,\chi/250)>0`}
  {@html eq`D_x=\frac{\partial F}{\partial v_x}=\underbrace{\frac{\partial I_{\text{ion}}}{\partial x}}_{\text{電流の係数}}\cdot\underbrace{\bar x'(v_x)}_{\text{定常値の傾き}}`}
  <p>電流の係数には、同じ電流に入るほかのゲートの今の値が入る。全状態が {@html tex`V`} で定常な点（{@html tex`v_x=V`}）で、各ゲートの {@html tex`D_x`} を計算する。</p>
  <Card title="感度 D の値" subtitle="全状態が V で定常な点での D_x = (∂I/∂x) x̄′(V)">
    <LineChart traces={sensitivityValues} xDomain={voltageDomain} xLabel={voltageLabel} yLabel="D (mS/cm²)" guides={[restLines, zeroLine]} />
    {#snippet info()}
      <p>ゲートと Ca 系を電位 {@html tex`V`} での定常値に置き、電流の係数を計算する。{@html tex`\bar x'`} は幅 ±10⁻⁴ mV の中心差分で求める（{@html tex`c`} だけは解析的に求める）。発火中の軌道の上の値ではない。</p>
    {/snippet}
  </Card>

  <h3>感度の符号</h3>
  <p>電流の係数は、電流の表の式をゲートで微分したもので、「正のコンダクタンス × ほかのゲート × {@html tex`(V-E_x)`}」の形をしている。だから {@html tex`D_x`} の符号は次の積で決まる。</p>
  {@html eq`\operatorname{sign}D_x=\operatorname{sign}(V-E_x)\cdot\operatorname{sign}\bar x'(v_x)`}
  <p>式から読むと、{@html tex`E_{\text K}<V<E_{\text{Na}}`} で次のとおりになる。</p>
  <table>
    <thead><tr><th>{@html tex`x`}</th><th>電流の係数 {@html tex`\partial I_{\text{ion}}/\partial x`}</th><th>{@html tex`\operatorname{sign}(V-E_x)`}</th><th>{@html tex`\operatorname{sign}\bar x'`}</th><th>{@html tex`\operatorname{sign}D_x`}</th></tr></thead>
    <tbody>
      <tr><td>{@html tex`m`}</td><td>{@html tex`2g_{\text{Na}}mh(V-E_{\text{Na}})`}</td><td>{@html tex`-`}</td><td>{@html tex`+`}</td><td>{@html tex`-`}</td></tr>
      <tr><td>{@html tex`h`}</td><td>{@html tex`g_{\text{Na}}m^2(V-E_{\text{Na}})`}</td><td>{@html tex`-`}</td><td>{@html tex`-`}</td><td>{@html tex`+`}</td></tr>
      <tr><td>{@html tex`s`}</td><td>{@html tex`2g_{\text{Ca}}sr(V-E_{\text{Ca}})`}</td><td>{@html tex`-`}</td><td>{@html tex`+`}</td><td>{@html tex`-`}</td></tr>
      <tr><td>{@html tex`r`}</td><td>{@html tex`g_{\text{Ca}}s^2(V-E_{\text{Ca}})`}</td><td>{@html tex`-`}</td><td>{@html tex`-`}（{@html tex`v_r>-60`}）<br />{@html tex`0`}（{@html tex`v_r\le-60`}）</td><td>{@html tex`+`}（{@html tex`v_r>-60`}）<br />{@html tex`0`}（{@html tex`v_r\le-60`}）</td></tr>
      <tr><td>{@html tex`n`}</td><td>{@html tex`g_{\text{DR}}(V-E_{\text K})`}</td><td>{@html tex`+`}</td><td>{@html tex`+`}</td><td>{@html tex`+`}</td></tr>
      <tr><td>{@html tex`a`}</td><td>{@html tex`g_{\text A}b(V-E_{\text K})`}</td><td>{@html tex`+`}</td><td>{@html tex`+`}</td><td>{@html tex`+`}</td></tr>
      <tr><td>{@html tex`b`}</td><td>{@html tex`g_{\text A}a(V-E_{\text K})`}</td><td>{@html tex`+`}</td><td>{@html tex`-`}</td><td>{@html tex`-`}</td></tr>
      <tr><td>{@html tex`c`}</td><td>{@html tex`g_{\text C}\min(1,\chi/250)(V-E_{\text K})`}</td><td>{@html tex`+`}</td><td>{@html tex`+`}（{@html tex`v_c\le-10`}）<br />{@html tex`0`}（{@html tex`v_c>-10`}）</td><td>{@html tex`+`}（{@html tex`v_c\le-10`}）<br />{@html tex`0`}（{@html tex`v_c>-10`}）</td></tr>
    </tbody>
  </table>
  <p>符号は表のとおりで、{@html tex`D`} は {@html tex`h,n,a`} で正、{@html tex`r,c`} で正か 0（{@html tex`r`} は {@html tex`v_r\le-60`}、{@html tex`c`} は {@html tex`v_c>-10`} で 0）、{@html tex`m,s,b`} で負である。{@html tex`D=0`} のゲートは重みが 0 になるだけで、ほかの重みは 0 と 1 の間に収まる。ψ 群は符号のそろうゲート同士（Kepler 規則 3）、φ 群は {@html tex`D_\mu<0`} のゲート（Kepler 規則 2）でしか作れないので、Traub では ψ 群は {@html tex`h,n,a,r,c`} 同士か {@html tex`m,s,b`} 同士、φ 群に入れられるのは {@html tex`m,s,b`} だけである。</p>

  <h2>近似のよさ</h2>
  <p>近似のよさは、φ 群を Kepler 式(11)、ψ 群を Kepler 式(10)で見る（p.384）。</p>

  <h3>平衡点</h3>
  <p>判定量を読む点として、平衡点を求める。一定の外部電流 {@html tex`I_0`} のもとで、ゲートと Ca 系を電位 {@html tex`V`} での定常値に置いた膜電流を、定常 I-V 曲線 {@html tex`I_{\text{ss}}`} と呼ぶ。</p>
  {@html eq`x=\bar x(V), \quad \dot\chi=\dot q=0, \quad I_{\text{ss}}(V) := I_{\text{ion}}\big(V,\{\bar x(V)\},q(V),\chi(V)\big) = I_0`}
  <p>平衡点は {@html tex`I_{\text{ss}}(V)=I_0`} の根である。{@html tex`I_0=0`} の根のうち、系が留まる安定なものが静止電位である。</p>
  <Card title="定常 I-V 曲線と平衡点" moreLabel="縦軸を広げて全体を見る" subtitle={`I₀ = 0 の根：${rest.map(v => `${v.toFixed(1)} mV`).join('、')}（左端が静止電位）`}>
    <figure>
      <figcaption>根のまわり <small>−75〜−25 mV、±2 µA/cm² に絞り、根で横切る向きを見る</small></figcaption>
      <LineChart traces={[{ label: '定常 I-V 曲線', points: steadyCurve }]} xDomain={[-75, -25]} yDomain={[-2, 2]} xLabel={voltageLabel} yLabel="I_ss (µA/cm²)" guides={[restLines, currentLine]} />
    </figure>
    {#snippet more()}
      <figure>
        <figcaption>全体 <small>−40 mV 付近に深い谷がある</small></figcaption>
        <LineChart traces={[{ label: '定常 I-V 曲線', points: steadyCurve }]} xDomain={voltageDomain} yDomain={[-160, 40]} xLabel={voltageLabel} yLabel="I_ss (µA/cm²)" guides={[restLines, currentLine]} />
      </figure>
    {/snippet}
    {#snippet info()}
      {@html eq`C\dot{V}=I_0-I_{ss}(V)`}
      <p>ゲートと Ca 系を電位 {@html tex`V`} での定常値に置くと、膜の式は上の形になる。縦線は {@html tex`I_0=0`} の 3 つの根を示す。</p>
      <p>曲線が右上がりの根では、上にずれると {@html tex`I_{ss}>I_0`} で {@html tex`\dot{V}<0`} となって下へ戻り、下にずれると上へ戻る。右下がりの根ではずれが広がる。左の根（静止電位）は右上がりなので戻り、中の根は右下がりなので離れる（発火の閾値）。右の根も右上がりだが、ゲートは瞬時には追いつかないので、11 状態系では留まらない。</p>
      <p>各根から ±0.5 mV ずらして 11 状態系を 1000 ms 積分すると、元の電位に戻るのは左だけである。式(10)は、この静止電位で読む。</p>
    {/snippet}
  </Card>

  <h3>式(11) φ 群の速さ</h3>
  {@html eq`\Big|\frac{1}{Ck_\mu}D_\mu\Big|\ll1 \quad (\mu\in\Phi) \qquad \text{Kepler 式(11)}`}
  <p>この値が小さいゲートだけを φ 群に入れられる。小さくなければ、{@html tex`\alpha_0`} を式(9)で解いても縮約は成り立たない。</p>
  <Card title="式(11) の判定">
    <figure>
      <figcaption>判定量：式(11)の左辺 <small>縦軸は対数</small></figcaption>
      <LineChart traces={ratios} xDomain={voltageDomain} yDomain={[1e-5, 100]} yScale="log" xLabel={voltageLabel} yLabel="判定量" guides={[unitLine]} />
    </figure>
    {#snippet info()}
      <p>{@html tex`D_x=(\partial I/\partial x)\,\bar{x}'(v_x)`}。{@html tex`\bar{x}'`} は幅 ±10⁻⁴ mV の中心差分で求める（{@html tex`c`} だけは解析的に求める）。式(11)の {@html tex`k_\mu`} は、展開の中心 {@html tex`v_\mu=V=\phi`} で {@html tex`\partial f_\mu/\partial v_\mu=-k_\mu`} と置いたもので、平衡点でなくても成り立つ。この図は比較のため、全状態が {@html tex`V`} で定常な点での値を {@html tex`V`} の関数として描く。発火中の軌道の上の値ではない。</p>
    {/snippet}
  </Card>

  <h3>式(10) ψ 群の速さ</h3>
  {@html eq`\Big|\sum_{j\in\Psi_k}\alpha_jk_j-k_i\Big|\,\Big[\sum_{j\in\Psi_k}k_j\Big]^{-1}\ll1 \quad (i\in\Psi_k) \qquad \text{Kepler 式(10)}`}
  <p>{@html tex`\alpha_j=D_j/\sum_{l\in\Psi_k}D_l`} は ψ 群の重み（Kepler 式(6)）である。この式は平衡点で評価する。平衡点では {@html tex`v_x=V`} だから、左辺は {@html tex`V`} だけで決まり、静止電位で読む。</p>

  <Card title="式(10) の判定" subtitle={atRest ? `静止電位 ${rest[0].toFixed(1)} mV で、判定量の最大は ${largest(atRest.consistency).toPrecision(2)}` : undefined}>
    {#snippet controls()}
      <Toggles label="ψ 群にまとめる組" bind:selected={recoveryShown} options={optionsOf(recoveryGates)} />
    {/snippet}
    {#if atRest}
      <table>
        <thead><tr><th>ゲート</th><th>判定量</th><th>重み α</th><th>D (mS/cm²)</th><th>k (ms⁻¹)</th></tr></thead>
        <tbody>
          {#each recovery as name, i (name)}
            <tr>
              <td>{name}</td>
              <td>{atRest.consistency[i].toPrecision(3)}</td>
              <td>{atRest.weights[i].toPrecision(3)}</td>
              <td>{sensitivity(name, rest[0]).toPrecision(3)}</td>
              <td>{speed(rates(rest[0])[name]).toPrecision(3)}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    {:else}
      <p class="note">ゲートを 2 つ以上選ぶと判定する。</p>
    {/if}
    {#snippet info()}
      <p>式(10)は平衡点で {@html tex`\partial f_i/\partial v_i=-k_i`} と置いて導いた式なので、静止電位（{@html tex`I_0=0`} の安定な平衡点）でだけ評価する。そこでは全ての等価電位が {@html tex`V`} に一致する。{@html tex`D_x=(\partial I/\partial x)\,\bar{x}'(v_x)`} の {@html tex`\bar{x}'`} は幅 ±10⁻⁴ mV の中心差分で求める（{@html tex`c`} だけは解析的に求める）。</p>
    {/snippet}
  </Card>
  <p>静止電位 −69 mV で、式(10)は {@html tex`\lbrace h,n\rbrace`} が 0.03 と小さく、{@html tex`\lbrace a,c\rbrace`} は 0.59 と大きい。{@html tex`b`} は膜電位よりずっと遅いので、φ 群に入れない。</p>

  <h2>縮約モデル</h2>
  <h3>群分けの規則</h3>
  <p>ここまでの計算から、Traub の 8 つのゲートの組み分けは次のように絞られる。</p>
  <table>
    <thead><tr><th>条件</th><th>分かったこと</th></tr></thead>
    <tbody>
      <tr><td>変換できる条件 {@html tex`\bar x'\neq0`}</td><td>{@html tex`r`} は {@html tex`v\le-60`}、{@html tex`c`} は {@html tex`v>-10`} で変換できない。群に入れても、そこでは重みが 0 になる。</td></tr>
      <tr><td>感度の符号</td><td>φ 群は {@html tex`D<0`} の {@html tex`m,s,b`} から選ぶ。ψ 群は {@html tex`h,n,a,r,c`} どうしか {@html tex`m,s,b`} どうしで作る。</td></tr>
      <tr><td>式(11)</td><td>{@html tex`b`} は膜電位よりずっと遅いので、φ 群は {@html tex`m,s`} から選ぶ。</td></tr>
      <tr><td>式(10)</td><td>静止電位で小さいのは {@html tex`\lbrace h,n\rbrace`}（0.03）だけ。{@html tex`a`} を加えると 0.2 前後、{@html tex`r`} か {@html tex`c`} を含むと 0.5 以上、{@html tex`m,s,b`} どうしは 0.5 以上になる。</td></tr>
    </tbody>
  </table>
  <p>{@html tex`b`} は φ 群にも ψ 群にも合わないので、元のまま残す。</p>

  <h3>ゲートの作り直し</h3>
  <p>群分けを決めると、Kepler 規則 4,5 で縮約モデルが一つに決まる。まず、各ゲートを属する群の代表値から作り直す。</p>
  {@html eq`x=\bar x(\phi)\quad(x\in\Phi), \qquad x=\bar x(\psi_k)\quad(x\in\Psi_k)`}
  <p>これを電流の表の {@html tex`I_{\text{ion}}`} に入れ、{@html tex`V`} を {@html tex`\phi`} に置き換えたものが縮約後の膜電流である。</p>
  {@html eq`F_{\text{red}}(z)=I_{\text{ion}}\big(\phi,\ \lbrace\bar x(\phi)\rbrace_{x\in\Phi},\ \lbrace\bar x(\psi_k)\rbrace_{x\in\Psi_k},\ q,\ \chi\big)`}

  <h3>重み</h3>
  <p>ψ 群の重みは群の感度の比である（Kepler 式(6)）。</p>
  {@html eq`\alpha_i=\frac{D_i}{\sum_{j\in\Psi_k}D_j}`}
  <p>φ 群で縮約式に残るのは {@html tex`\alpha_0`} だけで、Kepler 式(9)の根として決まる。φ 群のゲートは何個あっても {@html tex`\bar x_\mu(\phi)`} で作り直すので、ほかの重み {@html tex`\alpha_\mu`}（Kepler 式(8)）は使わない（Kepler 規則 4）。{@html tex`\nu`} は {@html tex`0`} と {@html tex`\Phi`} を走る。</p>
  {@html eq`\alpha_0\sum_\nu D_\nu-D_0-C\sum_{\mu\in\Phi}\frac{k_\mu D_\mu}{Ck_\mu-\alpha_0\sum_\nu D_\nu}=0`}
  <p>式(9)は {@html tex`\alpha_0`} について解が複数ある。そのうち、φ 群のゲートが十分速い（{@html tex`k_\mu\to\infty`}）とき {@html tex`\alpha_0\to1`}、つまり φ が膜電位そのものに近づく解を使う。{@html tex`\Phi=\lbrace\mu\rbrace`} が一つなら式(9)は二次方程式で、その解は</p>
  {@html eq`\alpha_0=\frac{2Ck_\mu}{(Ck_\mu+D_0)+\sqrt{(Ck_\mu-D_0)^2-4Ck_\mu D_\mu}}`}
  <p>となる。{@html tex`\Phi`} が空なら式(9)は {@html tex`\alpha_0D_0=D_0`} で {@html tex`\alpha_0=1`} である。Kepler は {@html tex`\alpha_0`} を式(9)から計算する。式(11)が成り立つ φ 群では {@html tex`\alpha_0`} は 1 に近いので、{@html tex`\alpha_0=1`} と置く近似もとれる。</p>
  <p>感度は各時刻に {@html tex`V=\phi`}、{@html tex`v_\mu=\phi`}、{@html tex`v_i=\psi_k`} で評価する。</p>

  <h3>運動方程式</h3>
  <p>状態 {@html tex`z=(\phi,\psi_1,\dots,\psi_K,\chi,q)`} は、群ごとに一本ずつの式で動く。</p>
  {@html eq`C\dot\phi=\alpha_0\,[I_{\text{ext}}(t)-F_{\text{red}}(z)], \qquad \dot\psi_k=\sum_{i\in\Psi_k}\alpha_i\,\frac{k_i(\phi)\,[\bar x_i(\phi)-\bar x_i(\psi_k)]}{\bar x_i'(\psi_k)}`}
  <p>{@html tex`\chi,q`} は {@html tex`\dot\chi=-\kappa I_{\text{Ca}}-0.075\chi`}、{@html tex`\dot q=\min(0.00002\chi,0.01)(1-q)-0.001q`} のままで、{@html tex`I_{\text{Ca}}`} には作り直した {@html tex`s,r`} と {@html tex`\phi`} を入れる。</p>
  <p>{@html tex`\alpha_0=1`} なら、膜電位の式は次のとおりである。</p>
  {@html eq`C\dot\phi=I_{\text{ext}}(t)-F_{\text{red}}(z)`}
</Page>
