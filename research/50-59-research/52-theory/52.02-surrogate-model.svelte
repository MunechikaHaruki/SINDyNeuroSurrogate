<script lang="ts">
  import { eq, tex } from '$lib/tex.ts';
  import Card from '$lib/Card.svelte';
  import Choice from '$lib/Choice.svelte';
  import LineChart from '$lib/LineChart.svelte';
  import MatrixTable from '$lib/MatrixTable.svelte';
  import Mermaid from '$lib/Mermaid.svelte';
  import Page from '$lib/Page.svelte';
  import Section from '$lib/Section.svelte';
  import Range from '$lib/Range.svelte';

  type Point = { x: number; y: number };
  type Gate = 'm' | 'h' | 'n' | 'a' | 'b' | 'c' | 's' | 'r';

  /** Traub19 の区画数。 */
  const compartments = 19;
  /** 元の単一区画の状態数：V、8 ゲート、q、χ。 */
  const fullStates = 11;
  /** 物理のまま解く Ca 系の状態：s, r, q, χ。 */
  const calciumStates = 4;
  /** 研究コードの AE の隠れ層の幅（preprocessor/autoencoder.py）。 */
  const hidden = 16;

  // ---- 区画の状態とメモリ ----

  type Precision = 'f64' | 'f32';
  const precisionOptions: { value: Precision; label: string }[] = [
    { value: 'f64', label: 'float64（8 バイト）' },
    { value: 'f32', label: 'float32（4 バイト）' },
  ];
  let precision = $state<Precision>('f64');
  let statesPerCompartment = $state(fullStates);
  let cacheKilobytes = $state(128);

  /** 1 状態のバイト数。 */
  function bytesOf(choice: Precision): number {
    if (choice === 'f64') return 8;
    return 4;
  }
  /** 1 細胞の状態が占めるバイト数 = 状態数 × 区画数 × バイト。 */
  function cellBytes(states: number, choice: Precision): number {
    return states * compartments * bytesOf(choice);
  }
  /** 容量 kilobytes の L1 に、状態だけで何細胞が載るか。 */
  function cellsInCache(states: number, choice: Precision, kilobytes: number): number {
    return Math.floor((kilobytes * 1024) / cellBytes(states, choice));
  }
  /** 区画あたりの状態数を 1〜11 と動かしたときの、L1 に載る細胞数。 */
  function cacheCurve(choice: Precision, kilobytes: number): Point[] {
    const points: Point[] = [];
    for (let states = 1; states <= fullStates; states++) points.push({ x: states, y: cellsInCache(states, choice, kilobytes) });
    return points;
  }

  // ---- 全ゲートをまとめて圧縮するモデル（NEURO2026） ----

  /** SINDy の項の数。レート項が圧縮前のゲートごとに α, β, α·z_j, β·z_j、基底が V, z_j, 1。 */
  function libraryTerms(rateGates: number, latents: number): number {
    const rateTerms = 2 * rateGates;
    const productTerms = 2 * rateGates * latents;
    const basisTerms = latents + 2;
    return rateTerms + productTerms + basisTerms;
  }
  /** AE の重みの数。encoder は gates → 16 → latents、decoder は latents → 16 → gates で、それぞれ重みとバイアスを持つ。 */
  function autoencoderWeights(gates: number, latents: number): number {
    const encoder = gates * hidden + hidden + hidden * latents + latents;
    const decoder = latents * hidden + hidden + hidden * gates + gates;
    return encoder + decoder;
  }
  /** NEURO2026 の構成：6 ゲートを潜在 5 本へ、レート項は 8 ゲートぶん。 */
  const posterTerms = libraryTerms(8, 5);

  const posterStep = `flowchart LR
    z["潜在 z（5 本）"] -- decoder --> x["6 ゲート<br/>m n c a h b"]
    ca["s r q χ<br/>物理のまま"] --> I["イオン電流 I_ion"]
    x --> I
    I --> dV["dV/dt"]
    z --> theta["Θ(z, V)"]
    dV -. V .-> theta
    theta -- Ξ --> dz["dz/dt"]
  `;

  // ---- 群ごとに圧縮するモデル ----

  type Grouping = 'hn' | 'hac' | 'all';
  const groupingOptions: { value: Grouping; label: string }[] = [
    { value: 'hn', label: '{h,n}{a,c}{b}' },
    { value: 'hac', label: '{h,a,c}{n}{b}' },
    { value: 'all', label: '6 ゲートを 1 群（NEURO2026）' },
  ];
  type GateMode = 'phi' | 'gate';
  const mOptions: { value: GateMode; label: string }[] = [
    { value: 'phi', label: '膜電位へ吸収（m = m̄(V)）' },
    { value: 'gate', label: '元のゲートの式のまま' },
  ];
  type SingleMode = 'gate' | 'learn';
  const singleOptions: { value: SingleMode; label: string }[] = [
    { value: 'gate', label: '元のゲートの式のまま' },
    { value: 'learn', label: '1 次元の AE + SINDy' },
  ];
  let grouping = $state<Grouping>('hn');
  let mMode = $state<GateMode>('phi');
  let singleMode = $state<SingleMode>('gate');
  let groupLatents = $state(1);
  let allLatents = $state(5);

  /** 学習する群一つ。gates は圧縮するゲート、rateGates は SINDy の項に使うレートのゲート、latents は潜在次元。 */
  type Block = { gates: Gate[]; rateGates: Gate[]; latents: number };
  /** 群分けの中身：学習する群と、元の式のまま解くゲート。 */
  type Plan = { blocks: Block[]; kept: Gate[]; absorbed: Gate[] };

  /** 2 つ以上のゲートの群と、一つだけの群を、選んだ組から並べる。 */
  function groupsOf(choice: Grouping): { multi: Gate[][]; single: Gate[] } {
    if (choice === 'hn')
      return {
        multi: [
          ['h', 'n'],
          ['a', 'c'],
        ],
        single: ['b'],
      };
    if (choice === 'hac') return { multi: [['h', 'a', 'c']], single: ['n', 'b'] };
    return { multi: [['m', 'h', 'n', 'a', 'b', 'c']], single: [] };
  }
  /** 選んだ組・m の扱い・一つだけの群の扱い・潜在次元から、学習する群と元の式で解くゲートを決める。 */
  function planOf(choice: Grouping, m: GateMode, single: SingleMode, latents: number, posterLatents: number): Plan {
    const groups = groupsOf(choice);
    const blocks: Block[] = [];
    const kept: Gate[] = [];
    const absorbed: Gate[] = [];
    if (choice === 'all') {
      // 研究コードの SINDy の項は、Ca 系の s, r を含む 8 ゲートのレートを並べている。
      blocks.push({ gates: groups.multi[0], rateGates: ['m', 'h', 'n', 'a', 'b', 'c', 's', 'r'], latents: posterLatents });
      return { blocks, kept, absorbed };
    }
    if (m === 'phi') absorbed.push('m');
    else kept.push('m');
    for (const gates of groups.multi) blocks.push({ gates, rateGates: gates, latents: Math.min(latents, gates.length) });
    for (const gate of groups.single) {
      if (single === 'learn') blocks.push({ gates: [gate], rateGates: [gate], latents: 1 });
      else kept.push(gate);
    }
    return { blocks, kept, absorbed };
  }
  let plan = $derived(planOf(grouping, mMode, singleMode, groupLatents, allLatents));

  /** 区画あたりの状態数 = V + 潜在 + 元の式のゲート + Ca 系 4 つ。 */
  function planStates(chosen: Plan): number {
    let states = 1 + chosen.kept.length + calciumStates;
    for (const block of chosen.blocks) states += block.latents;
    return states;
  }
  /** 群の係数 Ξ_k の数 = 潜在次元 × 項の数。 */
  function blockCoefficients(block: Block): number {
    return block.latents * libraryTerms(block.rateGates.length, block.latents);
  }
  /** 全細胞で共有する学習済みの数（Ξ の係数と AE の重み）。 */
  function sharedNumbers(chosen: Plan): number {
    let total = 0;
    for (const block of chosen.blocks) total += blockCoefficients(block) + autoencoderWeights(block.gates.length, block.latents);
    return total;
  }
  /** 群を「{h,n}」の形に書く。 */
  function blockLabel(block: Block): string {
    return `{${block.gates.join(',')}}`;
  }
  /** 群 k の潜在変数の見出し。1 本なら z_{hn}、2 本以上なら z_{hn,1}。 */
  function latentLabels(block: Block): string[] {
    const name = block.gates.length === 6 ? '' : block.gates.join('');
    const labels: string[] = [];
    for (let l = 1; l <= block.latents; l++) {
      if (name === '') labels.push(tex`z_{${l}}`);
      else if (block.latents === 1) labels.push(tex`z_{${name}}`);
      else labels.push(tex`z_{${name},${l}}`);
    }
    return labels;
  }

  /** 係数行列の行：レートの出どころのゲート 8 つと、基底の 2 行。 */
  const sourceGates: Gate[] = ['m', 'h', 'n', 'a', 'b', 'c', 's', 'r'];
  const sourceRows: string[] = [];
  for (const gate of sourceGates) sourceRows.push(tex`\alpha_${gate},\beta_${gate}`);
  sourceRows.push(tex`V,\ 1`);
  sourceRows.push(tex`z`);
  /** 列ごとに、その潜在変数の式が使える項の数を、出どころの行に入れる。0 は使えない項。 */
  function structureMatrix(chosen: Plan): { columns: string[]; values: number[][] } {
    const columns: string[] = [];
    const values: number[][] = [];
    for (let i = 0; i < sourceRows.length; i++) values.push([]);
    for (const block of chosen.blocks) {
      for (const label of latentLabels(block)) {
        columns.push(label);
        for (let i = 0; i < sourceGates.length; i++) {
          const used = block.rateGates.includes(sourceGates[i]);
          values[i].push(used ? 2 + 2 * block.latents : 0);
        }
        values[sourceGates.length].push(2);
        values[sourceGates.length + 1].push(block.latents);
      }
    }
    return { columns, values };
  }
  let structure = $derived(structureMatrix(plan));

  const groupStep = `flowchart LR
    subgraph k["群 k ごと"]
      zk["潜在 z_k"] -- "decoder D_k" --> xk["群のゲート"]
      zk --> thk["Θ_k(z_k, V)"]
      thk -- "Ξ_k" --> dzk["dz_k/dt"]
    end
    kept["元の式のゲート<br/>s r q χ"] --> I["イオン電流 I_ion"]
    xk --> I
    I --> dV["dV/dt"]
    dV -. V .-> thk
  `;
</script>

<Page title="Traub 区画のサロゲートの数理構成" summary="Traub の区画の状態を減らしてモデルを L1 キャッシュに収めるための、サロゲートの数理の構成。物理として残す部分、全ゲートをまとめて圧縮する AE + SINDy、KAM 縮約、その群分けで群ごとに圧縮するモデル、Multi-Compartment への適用、比べ方。">
  <p>Traub の 19 区画モデルを、状態を減らしたサロゲートに置き換える。狙いは計算量よりメモリで、1 細胞のモデルを GPU の L1 キャッシュに収める。この冊は、何を物理のまま残し、何をどう学習して、どの順に広げるかを、式の形で並べる。</p>

  <Section id="sec-memory" title="区画の状態とメモリ">
    <p>単一区画の状態は膜電位 {@html tex`V`}、ゲート 8 つ（{@html tex`m,h,n,a,b,c,s,r`}）、AHP ゲート {@html tex`q`}、Ca 濃度に比例する {@html tex`\chi`} の 11 個である。式は <a href="../../../../../resources/textbook/40-49-neuroscience/42-neuron-models/42.07-traub-model">別冊「Traub モデル」</a>にある。</p>
    {@html eq`\underbrace{V}_{1},\ \underbrace{m,h,n,a,b,c,s,r}_{8},\ \underbrace{q,\ \chi}_{2}\qquad 11\times19=209\ \text{状態/細胞}`}
    <p>時間積分は刻みごとに全状態を読んで書く。細胞を数千並べると、1 刻みに動かすバイト数は細胞数に比例し、計算より先にメモリの帯域で詰まる。1 細胞の状態が占めるバイト数は次で決まる。</p>
    {@html eq`\text{1 細胞のバイト数}=\underbrace{(\text{区画あたりの状態数})}_{\text{サロゲートで減らす}}\times\underbrace{19}_{\text{区画}}\times\underbrace{(\text{1 状態のバイト数})}_{\text{精度}}`}
    <p>区画あたりの状態数を減らすのがこの研究の手段で、残りの二つは別の手段になる。</p>

    <Card kicker="メモリ" title="L1 に載る細胞の数" subtitle={`${statesPerCompartment} 状態 × 19 区画 × ${bytesOf(precision)} バイト = ${cellBytes(statesPerCompartment, precision)} バイト/細胞、L1 ${cacheKilobytes} KB に ${cellsInCache(statesPerCompartment, precision, cacheKilobytes)} 細胞`}>
      {#snippet controls()}
        <Range label="区画あたりの状態数" bind:value={statesPerCompartment} min={1} max={fullStates} step={1} display={`${statesPerCompartment}`} />
        <Choice label="精度" bind:value={precision} options={precisionOptions} />
        <Range label="L1 の容量" bind:value={cacheKilobytes} min={32} max={256} step={32} display={`${cacheKilobytes} KB`} />
      {/snippet}
      <LineChart traces={[{ label: 'L1 に載る細胞数', points: cacheCurve(precision, cacheKilobytes) }]} xDomain={[1, fullStates]} xLabel="区画あたりの状態数" yLabel="細胞数" guides={[{ label: '選んだ状態数', tone: 'accent', x: [statesPerCompartment] }]} />
      {#snippet info()}
        <p>状態だけを数える。区画ごとのコンダクタンス密度などのパラメータと、学習した係数・重みは別に要る。学習した係数・重みは全細胞で共有するので、細胞数には比例しない。L1 の容量は GPU の世代で違うので、つまみで変える。</p>
      {/snippet}
    </Card>
  </Section>
  <Section id="sec-physics" title="物理として残す部分">
    <p>サロゲートが学習するのは、電位だけで動く 6 ゲート {@html tex`m,n,c,a,h,b`} の時間発展だけである。膜電位の式と Ca 系は物理の式のまま解く。</p>
    {@html eq`C\frac{dV}{dt}=I_{\mathrm{ext}}-\underbrace{I_{\mathrm{ion}}(V;\ m,h,n,a,b,c;\ s,r,q,\chi)}_{\text{物理の式}}`}
    {@html eq`\frac{d\chi}{dt}=-\kappa\,I_{\mathrm{Ca}}(V,s,r)-0.075\,\chi,\qquad \frac{dq}{dt}=\min(2\times10^{-5}\chi,\,0.01)(1-q)-0.001\,q`}
    <p>膜電位を残すので、区画間の結合（軸方向の電流）はそのまま足せる。s, r を学習側に入れると、{@html tex`I_{\mathrm{Ca}}=g_{\mathrm{Ca}}s^2r(V-E_{\mathrm{Ca}})`} の誤差が {@html tex`s`} の 2 乗で増え、積分器である {@html tex`\chi`} に溜まる。そのため s, r も物理側に回す。</p>
    <p>コンダクタンス密度 {@html tex`g`} は {@html tex`I_{\mathrm{ion}}`} にしか入らない。学習するゲートの式は {@html tex`g`} を含まないので、密度の違う区画にも同じサロゲートを当てられる（<a href="#sec-mc">Multi-Compartment への適用</a>）。</p>
  </Section>
  <Section id="sec-poster" title="全ゲートをまとめて圧縮するモデル">
    <p>NEURO2026 で発表した形。6 ゲートを AutoEncoder で潜在変数 {@html tex`z`} に縮め、{@html tex`z`} の時間発展を SINDy で式として同定する（<a href="https://doi.org/10.1073/pnas.1906995116">Champion et al. 2019</a>）。</p>
    {@html eq`z=E(x)\in\mathbb{R}^{d},\qquad \hat{x}=D(z)\in\mathbb{R}^{6},\qquad x=(m,n,c,a,h,b)`}
    <p>{@html tex`E`} と {@html tex`D`} は隠れ層 16 の tanh の層を一つずつ持つ。{@html tex`E`} は学習と初期値にだけ使い、積分では {@html tex`D`} だけを毎刻み呼ぶ。潜在変数の式は、候補の項 {@html tex`\Theta`} の線形和で書く。</p>
    {@html eq`\frac{dz}{dt}=\Xi\,\Theta(z,V),\qquad \Theta=\big[\underbrace{\alpha_x(V),\ \beta_x(V)}_{\text{レート}},\ \underbrace{\alpha_x(V)z_j,\ \beta_x(V)z_j}_{\text{レート}\times\text{潜在}},\ V,\ z_j,\ 1\big]`}
    <p>項はゲートの式 {@html tex`dx/dt=\alpha_x(1-x)-\beta_x x`} に現れる形を潜在変数へ移したもので、{@html tex`x`} は s, r を含む 8 ゲートを走る。項の数はレートのゲート数 {@html tex`G`} と潜在次元 {@html tex`d`} で決まる。</p>
    {@html eq`|\Theta|=2G+2Gd+d+2,\qquad G=8,\ d=5\ \Rightarrow\ |\Theta|=${posterTerms},\quad \Xi\ \text{は}\ 5\times${posterTerms}`}
    <Mermaid label="NEURO2026 のモデルの 1 刻み" code={posterStep} />
    <div class="panel caution">
      <p><b>この形の限界</b></p>
      <p>6 → 5 より先で再現が崩れ、圧縮率が頭打ちになった。積分の毎刻みに decoder を通すので、減らした状態の代わりに計算が増える。同定した {@html tex`\Xi`} の係数の 79.6% が非零で、式として読めない。原因は、時定数の違うゲートを一つの潜在空間に混ぜていることだと見ている（確かめてはいない）。</p>
    </div>
  </Section>
  <Section id="sec-kam" title="KAM 縮約">
    <p>Kepler–Abbott–Marder 縮約は、時定数と電流への効き方がそろうゲートを、一つの変数にまとめる（<a href="https://doi.org/10.1007/BF00197717">Kepler et al. 1992</a>）。導出と、Traub で群を作れる条件は <a href="52.01-apply-kam">別冊「Traub 単一コンパートメントへの Kepler–Abbott–Marder 法の適用」</a>にある。ここでは縮約後の形だけを置く。</p>
    <p>膜より十分速い m は膜電位へ吸収し（φ 群）、残りは群 {@html tex`\Psi_k`} ごとに一つの等価電位 {@html tex`\psi_k`} で代表させる。</p>
    {@html eq`m=\bar{m}(V),\qquad x_i=\bar{x}_i(\psi_k)\quad(i\in\Psi_k)`}
    {@html eq`C\frac{dV}{dt}=\alpha_0\big[I_{\mathrm{ext}}-I_{\mathrm{ion}}\big],\qquad \frac{d\psi_k}{dt}=\sum_{i\in\Psi_k}\underbrace{\frac{D_i}{\sum_{j\in\Psi_k}D_j}}_{\text{重み}\ \alpha_i}\ \underbrace{\frac{k_i(V)\,[\bar{x}_i(V)-\bar{x}_i(\psi_k)]}{\bar{x}_i'(\psi_k)}}_{f_i}`}
    <p>{@html tex`D_i=(\partial I/\partial x_i)\,\bar{x}_i'`} は感度、{@html tex`k_i=\alpha_i+\beta_i`} はゲートの緩和の速さである。6 ゲートを 3 本にする組の候補は {@html tex`\{h,n\}\{a,c\}\{b\}`} と {@html tex`\{h,a,c\}\{n\}\{b\}`} で、どちらも区画あたり 11 → 8 状態になる。</p>
    {@html eq`V,\ \underbrace{\psi_1,\ \psi_2,\ b}_{\text{6 ゲートが 3 本}},\ s,\ r,\ q,\ \chi`}
    <p>KAM の式は学習を含まない。群の中の重み {@html tex`\alpha_i`} と右辺は、ゲートの式から毎刻み計算する。</p>
  </Section>
  <Section id="sec-grouped" title="群ごとに圧縮するモデル">
    <p>これから作るモデル。KAM の群分けで、時定数の近いゲートどうしだけを一つの小さな AE と SINDy に渡し、群の間では潜在変数を混ぜない。群 {@html tex`G_k`} ごとに次を持つ。</p>
    {@html eq`z_k=E_k(x_{G_k})\in\mathbb{R}^{d_k},\qquad \hat{x}_{G_k}=D_k(z_k),\qquad \frac{dz_k}{dt}=\Xi_k\,\Theta_k(z_k,V)`}
    {@html eq`\Theta_k=\big[\alpha_x(V),\ \beta_x(V),\ \alpha_x(V)z_{k,l},\ \beta_x(V)z_{k,l},\ V,\ z_{k,l},\ 1\big]_{x\in G_k}`}
    <p>{@html tex`\Theta_k`} のレートは群 {@html tex`G_k`} のゲートのものだけに限る。全体の係数は群ごとのブロックを対角に並べた形になり、時定数の違う群の項が式に入らない。</p>
    {@html eq`\Xi=\begin{pmatrix}\Xi_1&&\\&\Xi_2&\\&&\ddots\end{pmatrix}`}
    <p>{@html tex`d_k=1`} と置くと、状態の数は KAM 縮約と同じになる。違いは、まとめた座標 {@html tex`z_k`} とその右辺を、KAM は感度と速さから解析的に決め、このモデルはデータから学習する点だけである。二つを同じ群分けで比べられる。</p>
    <Mermaid label="群ごとに圧縮するモデルの 1 刻み" code={groupStep} />

    <Card kicker="構成" title="群分けと状態・係数の数" subtitle={`区画あたり ${planStates(plan)} 状態、1 細胞 ${cellBytes(planStates(plan), precision)} バイト、共有する係数と重み ${sharedNumbers(plan)} 個（${sharedNumbers(plan) * bytesOf(precision)} バイト）`} moreLabel="群ごとの内訳">
      {#snippet controls()}
        <Choice label="組" bind:value={grouping} options={groupingOptions} />
        <Choice label="m の扱い" bind:value={mMode} options={mOptions} disabled={grouping === 'all'} />
        <Choice label="一つだけの群の扱い" bind:value={singleMode} options={singleOptions} disabled={grouping === 'all'} />
        {#if grouping === 'all'}
          <Range label="潜在次元 d" bind:value={allLatents} min={1} max={6} step={1} display={`${allLatents}`} />
        {:else}
          <Range label="2 つ以上の群の潜在次元 d_k" bind:value={groupLatents} min={1} max={3} step={1} display={`${groupLatents}`} />
        {/if}
        <Choice label="精度" bind:value={precision} options={precisionOptions} />
      {/snippet}
      <figure>
        <figcaption>潜在変数の式が使える項 <small>行は項の出どころ、値は項の数、· は使わない</small></figcaption>
        <MatrixTable rows={sourceRows} columns={structure.columns} values={structure.values} />
      </figure>
      <p class="note">
        元の式のまま解くゲート：{plan.kept.length > 0 ? plan.kept.join(', ') : 'なし'}。膜電位へ吸収：{plan.absorbed.length > 0 ? plan.absorbed.join(', ') : 'なし'}。Ca 系 s, r, q, χ は常に物理のまま。
      </p>
      {#snippet more()}
        <table>
          <thead><tr><th>群</th><th>潜在次元</th><th>項の数</th><th>Ξ の係数</th><th>AE の重み</th></tr></thead>
          <tbody>
            {#each plan.blocks as block (blockLabel(block))}
              <tr>
                <td>{blockLabel(block)}</td>
                <td>{block.latents}</td>
                <td>{libraryTerms(block.rateGates.length, block.latents)}</td>
                <td>{blockCoefficients(block)}</td>
                <td>{autoencoderWeights(block.gates.length, block.latents)}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      {/snippet}
      {#snippet info()}
        <p>状態数は {@html tex`V`}、潜在変数、元の式のまま解くゲート、Ca 系の 4 つを足したもの。1 細胞のバイト数は 19 区画ぶん。係数と重みは学習した数で、全細胞・全区画で共有するので細胞数には比例しない。</p>
        <p>AE の隠れ層は研究コードと同じ 16 で数える。2 つ以上の群の潜在次元は、群のゲート数を超えない。「6 ゲートを 1 群」は NEURO2026 の構成で、項のレートは s, r を含む 8 ゲートぶんを並べる。</p>
      {/snippet}
    </Card>

    <div class="panel caution">
      <p><b>未決</b></p>
      <p>群の潜在次元 {@html tex`d_k`} を 1 に固定するか群ごとに選ぶか、m を膜電位へ吸収するか元の式で残すか、一つだけの群（b など）を学習するか元の式で解くか。6 ゲートを 3 本にする組（{@html tex`\{h,n\}\{a,c\}\{b\}`} か {@html tex`\{h,a,c\}\{n\}\{b\}`}）も未決で、<a href="../53-evaluation/53.01-group-comparison">別冊「Traub 単一区画：ゲートの組を選び、縮約と比べる」</a>で比べている。</p>
    </div>
  </Section>
  <Section id="sec-mc" title="Multi-Compartment への適用">
    <p>Traub 原著はチャネルの動き方を全区画で同じとし、区画で違うのはコンダクタンス密度だけである（<a href="https://doi.org/10.1152/jn.1991.66.2.635">Traub et al. 1991</a>）。密度は物理側の {@html tex`I_{\mathrm{ion}}`} にしか入らないので、全区画で共通のサロゲートを 1 つ作る。</p>
    {@html eq`C\frac{dV_c}{dt}=I_{\mathrm{ext},c}-I_{\mathrm{ion}}(V_c;\ \hat{x}_c;\ g_c)-\sum_{c'\sim c}\frac{V_c-V_{c'}}{R_{cc'}},\qquad \frac{dz_{k,c}}{dt}=\Xi_k\,\Theta_k(z_{k,c},V_c)`}
    <p>{@html tex`c`} は区画、{@html tex`g_c`} はその区画の密度、和は隣の区画 {@html tex`c'`} を走る。{@html tex`\Xi_k`} と AE は区画によらず一つで、区画ごとに持つのは状態だけである。</p>
    <ul>
      <li>学習データは 19 区画すべての軌道から取る。原著のシナプス入力は区画 3・15（樹状突起の中ほど）に入るので、そこへの入力も含める。</li>
      <li>群分けは区画ごとに確かめ直す。{@html tex`g_{\mathrm{Na}}=g_{\mathrm{DR}}=0`} の区画では h, n が電流に効かないので、群から除く。</li>
      <li>共通 1 つで再現できなければ、basal・soma・apical の 3 群に分けてサロゲートを作る。</li>
      <li>バーストは樹状突起の Ca スパイクが作るので、再現できたかは soma の波形だけで決めず、樹状突起の Ca 電流・{@html tex`\chi`}・AHP も見る。</li>
    </ul>
  </Section>
  <Section id="sec-compare" title="比べ方">
    <p>縮約の仕方（KAM、群ごとの AE + SINDy、全ゲートの AE + SINDy）と組み分けごとに、横軸に 1 細胞のメモリ、縦軸に忠実さをとって点を並べ、パレート前線で比べる。電位の RMSE と発火数は主にしない。</p>
    <ul>
      <li>忠実さの指標の候補：coincidence factor {@html tex`\Gamma`}（{@html tex`\Delta=2\text{〜}4`} ms）、f–I 曲線と発火の始まり、スパイクを除いた閾値下の RMSE、{@html tex`\chi`} と {@html tex`q`} の履歴。</li>
      <li>入力の候補：ステップ、過分極からの解放、OU 雑音。</li>
      <li>誤差は「縮約だけ → 粗い solver → 学習した補正」と段を分けて測る。</li>
    </ul>
    <p>メモリの壁が解けるかは、細胞を数千以上並べて GPU で回し、Nsight Compute（ncu）で測る。</p>
  </Section>
</Page>
