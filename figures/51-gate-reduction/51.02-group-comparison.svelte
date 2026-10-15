<script lang="ts">
  import { onMount } from 'svelte';
  import { eq, tex } from '$lib/tex.ts';
  import Dashboard from '$lib/Dashboard.svelte';
  import Card from '$lib/Card.svelte';
  import Range from '$lib/Range.svelte';
  import LineChart from '$lib/LineChart.svelte';
  import Toggles from '$lib/Toggles.svelte';
  import Choice from '$lib/Choice.svelte';
  import { checkTrajectory, duration, equilibria, equivalentPotential, groupCriteria, groupLayout, initial, input, psiPartitions, readFull, readSample, rmse, simulate, spikes, type Alpha0Mode, type Config, type GateName, type Grouping, type GroupCheck, type Partition, type Sample, type Verdict } from './traub.ts';

  type Point = { x: number; y: number };
  type Trace = { label: string; points: Point[]; dashed?: boolean };
  type Variable = 'V' | GateName;
  /** 選べる、電位だけで動く 6 ゲート。s,r は Ca 系と一緒に残すので選べない。 */
  const panelGates: GateName[] = ['m', 'h', 'n', 'a', 'b', 'c'];
  const variableOptions: { value: Variable; label: string }[] = [
    { value: 'V', label: 'V' },
    { value: 'm', label: 'm' },
    { value: 'h', label: 'h' },
    { value: 'n', label: 'n' },
    { value: 'a', label: 'a' },
    { value: 'b', label: 'b' },
    { value: 'c', label: 'c' },
  ];
  let shown = $state<Variable[]>(['h', 'n']);

  // ---- 縮約する群分け：φ 群を選び、残りを ψ 群に組み分ける ----
  /** φ 群に入れられるゲート。感度が負で、静止電位で式(11)が小さいもの（b は式(11)で外れる）。 */
  const phiCandidates: GateName[] = ['m'];
  let phiChosen = $state<GateName[]>(['m']);
  const alpha0Options: { value: Alpha0Mode; label: string }[] = [
    { value: 'one', label: '1 と置く（φ 群は速いゲートだけ）' },
    { value: 'solve', label: 'Kepler 式(9)を解く' },
  ];
  let alpha0 = $state<Alpha0Mode>('one');

  // ---- 式(10)：ψ 群を静止電位で判定する ----
  /** I₀ = 0 の安定な平衡点（静止電位）。式(10)は平衡点で導いた式なので、ここでだけ読む。 */
  const rest = equilibria(0)[0];
  /** 式(10)の左辺の目安。Kepler は値を定めていない。 */
  let epsilon = $state(0.6);
  let psiGates = $derived(panelGates.filter(name => !phiChosen.includes(name)));
  let allPartitions = $derived(psiPartitions(psiGates, rest));
  /** ε 以下の組み分け。式(10)の小さい順、同じなら変数の少ない順。 */
  let validPartitions = $derived(allPartitions.filter(p => p.worst <= epsilon).sort((x, y) => x.worst - y.worst || x.blocks.length - y.blocks.length));
  /** 既定は ψ{h,n}、ψ{a,c}、b 単独。選んだ組み分けが ε を超えたら、一覧の先頭に移る。 */
  let partitionKey = $state('h,n|a,c|b');
  let partition = $derived(validPartitions.find(p => p.key === partitionKey) ?? validPartitions[0]);
  $effect(() => {
    if (partition.key !== partitionKey) partitionKey = partition.key;
  });
  /** 組み分けを「ψ{h,n}、ψ{a,c}、b」の形に書く。 */
  function partitionLabel(p: Partition): string {
    return p.blocks.map(block => (block.length === 1 ? block[0] : `ψ{${block.join(',')}}`)).join('、');
  }
  let grouping = $derived<Grouping>({
    phi: panelGates.filter(name => phiChosen.includes(name)),
    psi: partition.blocks.filter(block => block.length >= 2),
    alpha0,
  });
  /** 2 つ以上のゲートが入った ψ 群ごとの、静止電位での式(10)の左辺。 */
  let psiChecks = $derived(grouping.psi.map(names => ({ names, ...groupCriteria(names, rest) })));

  /** 群分けを「Φ{m}、ψ{h,n}、ψ{a,c}、b」の形に書く。 */
  function groupingLabel(grouping: Grouping): string {
    const parts: string[] = [];
    if (grouping.phi.length > 0) parts.push(`Φ{${grouping.phi.join(',')}}`);
    for (const group of groupLayout(grouping)) {
      if (!panelGates.includes(group[0])) continue;
      parts.push(group.length === 1 ? group[0] : `ψ{${group.join(',')}}`);
    }
    if (grouping.phi.length > 0) parts.push(grouping.alpha0 === 'solve' ? 'α₀ は式(9)' : 'α₀ = 1');
    return parts.join('、');
  }
  /** 6 ゲートが縮約後に何本の変数になるか。 */
  function variableCount(grouping: Grouping): number {
    let count = 0;
    for (const group of groupLayout(grouping)) if (panelGates.includes(group[0])) count++;
    return count;
  }

  // ---- 時間の図：積分と表示範囲 ----
  let mean = $state(5),
    amplitude = $state(5);
  let windowStart = $state(0),
    windowLength = $state(100);
  let busy = $state(false),
    error = $state('');
  let result = $state<{ config: Config; grouping: Grouping; full: Sample[]; reduced: Sample[]; tolerance: number }>();
  let mounted = true;
  const pause = () => new Promise(resolve => setTimeout(resolve, 0));
  async function run(tolerance = 1e-7) {
    if (busy) return;
    const config = { mean, amplitude };
    const chosen = $state.snapshot(grouping);
    busy = true;
    error = '';
    await pause();
    try {
      const full = simulate(config, undefined, tolerance);
      await pause();
      if (!mounted) return;
      const reduced = simulate(config, chosen, tolerance);
      result = { config, grouping: chosen, full, reduced, tolerance };
    } catch (e) {
      result = undefined;
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }
  onMount(() => {
    mounted = true;
    void run();
    return () => {
      mounted = false;
    };
  });
  let domain = $derived<[number, number]>([windowStart, Math.min(duration, windowStart + windowLength)]);
  let pending = $derived(result && (mean !== result.config.mean || amplitude !== result.config.amplitude || groupingLabel(grouping) !== groupingLabel(result.grouping)));

  /** 元モデルの膜電位 V の時系列。 */
  function voltageSeries(samples: Sample[]): Point[] {
    const points: Point[] = [];
    for (const sample of samples) points.push({ x: sample.time, y: readFull(sample.state).v });
    return points;
  }
  /** 元モデルのゲート x の時系列（原著 図1）。 */
  function gateSeries(samples: Sample[], name: GateName): Point[] {
    const points: Point[] = [];
    for (const sample of samples) points.push({ x: sample.time, y: readFull(sample.state).gates[name] });
    return points;
  }
  /** 同じゲートを等価電位 v = x̄⁻¹(x) に直した時系列（原著 §2、図2）。 */
  function equivalentSeries(samples: Sample[], name: GateName): Point[] {
    const points: Point[] = [];
    for (const sample of samples) {
      const x = readFull(sample.state).gates[name];
      points.push({ x: sample.time, y: equivalentPotential(name, x) });
    }
    return points;
  }
  /** 6 ゲートの等価電位をまとめて計算する。逆関数を二分法で解くので、表示幅を動かすたびには計算しない。 */
  function equivalentTable(samples: Sample[]): Record<string, Point[]> {
    const table: Record<string, Point[]> = {};
    for (const name of panelGates) table[name] = equivalentSeries(samples, name);
    return table;
  }
  let equivalents = $derived(result ? equivalentTable(result.full) : undefined);
  /** 選んだゲートを panelGates の並びにそろえる。色が選んだ順で入れ替わらないようにする。 */
  function selectedGates(selection: Variable[]): GateName[] {
    const names: GateName[] = [];
    for (const name of panelGates) if (selection.includes(name)) names.push(name);
    return names;
  }
  let gates = $derived(selectedGates(shown));
  let withVoltage = $derived(shown.includes('V'));
  /** 図1：選んだゲートを 0〜1 の同じ軸に重ねる。 */
  function gateTraces(samples: Sample[], names: GateName[]): Trace[] {
    const list: Trace[] = [];
    for (const name of names) list.push({ label: name, points: gateSeries(samples, name) });
    return list;
  }
  /** 図2：選んだゲートの等価電位を重ねる。V を選んでいれば最後に足し、ゲートの色を図1とそろえる。 */
  function equivalentTraces(samples: Sample[], names: GateName[], voltage: boolean): Trace[] {
    const list: Trace[] = [];
    for (const name of names) list.push({ label: `v_${name}`, points: equivalents![name] });
    if (voltage) list.push({ label: 'V', points: voltageSeries(samples) });
    return list;
  }
  type Quantity = 'v' | 'q' | 'chi';
  /** 標本から膜電位・AHP ゲート・Ca 状態の列を取り出す。縮約系は群分けで 11 状態の形に読み直す。 */
  function valuesOf(samples: Sample[], chosen: Grouping | undefined, quantity: Quantity): number[] {
    const values: number[] = [];
    for (const sample of samples) values.push(readSample(sample, chosen)[quantity]);
    return values;
  }
  /** 元モデルと縮約の、同じ量の RMSE。 */
  function difference(quantity: Quantity): number {
    return rmse(valuesOf(result!.full, undefined, quantity), valuesOf(result!.reduced, result!.grouping, quantity));
  }
  /** 下段の比較：同じ量を 11 状態（実線）と縮約（破線）で重ねる。 */
  function traces(quantity: Quantity): Trace[] {
    if (!result) return [];
    const full: Point[] = [];
    const reduced: Point[] = [];
    const a = valuesOf(result.full, undefined, quantity);
    const b = valuesOf(result.reduced, result.grouping, quantity);
    for (let i = 0; i < a.length; i++) {
      full.push({ x: result.full[i].time, y: a[i] });
      reduced.push({ x: result.reduced[i].time, y: b[i] });
    }
    return [
      { label: '元モデル・11状態（実線）', points: full },
      { label: `縮約・${initial(result.grouping).length}状態（破線）`, dashed: true, points: reduced },
    ];
  }

  // ---- 時間の図：選んだ組を作れるか（51.01-apply-kam の条件 1,2） ----
  /** V を選べば膜電位側（φ）、選ばなければ回復側（ψ）として判定する。 */
  let judged = $derived(withVoltage ? gates.length >= 1 : gates.length >= 2);
  let groupCheck = $derived(result && judged ? checkTrajectory(gates, result.full, withVoltage) : undefined);
  const verdictLabels: Record<Verdict, string> = {
    ok: '満たす',
    noEquivalent: '等価電位なし（条件 1）',
    wrongSign: '符号が合わない（条件 2）',
    noWeight: '感度がすべて 0',
  };
  /** 判定ごとの時間を「満たす 500.1 ms、…」の形に並べる。0 ms の判定は省く。 */
  function verdictSummary(durations: Record<Verdict, number>): string {
    const parts: string[] = [];
    for (const verdict of Object.keys(verdictLabels) as Verdict[]) {
      if (durations[verdict] > 0) parts.push(`${verdictLabels[verdict]} ${durations[verdict].toFixed(1)} ms`);
    }
    return parts.join('、');
  }
  /** 満たさない判定ごとに、最初に起きた時刻（ms）。 */
  function firstFailures(samples: Sample[], checks: GroupCheck[]): string {
    const parts: string[] = [];
    for (const verdict of Object.keys(verdictLabels) as Verdict[]) {
      if (verdict === 'ok') continue;
      for (let j = 0; j < checks.length; j++) {
        if (checks[j].verdict !== verdict) continue;
        parts.push(`${verdictLabels[verdict]}：t = ${samples[j].time.toFixed(1)} ms`);
        break;
      }
    }
    return parts.join('、');
  }
  /** 軌道の上の重み α_i（ψ 群）か感度 D_μ（φ 群）を、ゲートごとに 1 本。 */
  function groupTraces(samples: Sample[], names: GateName[], voltage: boolean): Trace[] {
    const list: Trace[] = [];
    for (const name of names) list.push({ label: voltage ? `D・${name}` : `α・${name}`, points: [] });
    const checks = groupCheck!.checks;
    for (let j = 0; j < checks.length; j++) {
      const values = voltage ? checks[j].sensitivities : checks[j].weights;
      for (let i = 0; i < names.length; i++) list[i].points.push({ x: samples[j].time, y: values[i] });
    }
    return list;
  }

  const zeroLine = { label: '0', tone: 'muted' as const, y: [0] };
  const oneLine = { label: '1', tone: 'muted' as const, y: [1] };
</script>

<Dashboard title="Traub 単一区画：ゲートの組を選び、縮約と比べる" summary="元モデルの各変数と等価電位を並べてまとめる組を選び、その組を作れるか（等価電位と感度の符号）を元モデルの軌道の上で確かめる。最後に同じ入力で 11 状態と縮約した系を積分し、発火時刻・波形・Ca と AHP の履歴を比べる。" complete={true}>
  {#snippet sidebar()}
    <p class="note">導出と記号は <a href="51.01-apply-kam">別冊「Traub 単一コンパートメントへの Kepler–Abbott–Marder 法の適用」</a> にある。</p>
    <section>
      <h3>入力</h3>
      <Range label="平均電流密度 I₀" bind:value={mean} min={0} max={20} step={0.5} display={`${mean} µA/cm²`} disabled={busy} />
      <Range label="入力振幅 A" bind:value={amplitude} min={0} max={20} step={0.5} display={`${amplitude} µA/cm²`} disabled={busy} />
      <button type="button" onclick={() => void run()} disabled={busy}>{busy ? '計算中…' : 'この入力で計算'}</button>
      <button type="button" onclick={() => void run(1e-9)} disabled={busy}>精度を上げて計算</button>
      <p class="note" aria-live="polite">
        {#if busy}計算しています。
        {:else if error}停止：{error}
        {:else if result}表示中 I₀ = {result.config.mean}、A = {result.config.amplitude}、相対許容誤差 {result.tolerance.toExponential(0)}。{#if pending}<br />設定が変わりました。計算で反映します。{/if}
        {/if}
      </p>
    </section>
    <section>
      <h3>時間の表示</h3>
      <Range label="表示開始" bind:value={windowStart} min={0} max={450} step={10} display={`${windowStart} ms`} />
      <Range label="表示幅" bind:value={windowLength} min={50} max={500} step={50} display={`${Math.min(windowLength, duration - windowStart)} ms`} />
    </section>
  {/snippet}

  {#snippet left()}
    <Card kicker="静止電位" title="群分けと式(10) の判定" subtitle={`${groupingLabel(grouping)}：6 ゲート → ${variableCount(grouping)} 変数、全体で ${initial(grouping).length} 状態`} moreLabel="選んだ組み分けの、ゲートごとの判定量">
      {#snippet controls()}
        <Toggles label="φ 群（膜電位と一緒にまとめる）" bind:selected={phiChosen} options={phiCandidates.map(name => ({ value: name, label: name }))} disabled={busy} />
        <Choice label="φ 群の重み α₀" bind:value={alpha0} options={alpha0Options} disabled={busy || grouping.phi.length === 0} />
        <Range label="式(10) の目安 ε" bind:value={epsilon} min={0.01} max={1} step={0.01} display={epsilon.toFixed(2)} disabled={busy} />
      {/snippet}
      <p class="note">残りの {psiGates.join(', ')} を ψ 群に組み分ける。静止電位での式(10)が ε 以下のもの {validPartitions.length} 通りを、式(10)の小さい順に並べる。</p>
      <table>
        <thead><tr><th>選ぶ</th><th>組み分け</th><th>変数</th><th>式(10)</th></tr></thead>
        <tbody>
          {#each validPartitions as p (p.key)}
            <tr>
              <td><input type="radio" name="partition" value={p.key} bind:group={partitionKey} disabled={busy} aria-label={partitionLabel(p)} /></td>
              <td>{partitionLabel(p)}</td>
              <td>{p.blocks.length}</td>
              <td>{p.worst > 0 ? p.worst.toPrecision(2) : '—'}</td>
            </tr>
          {/each}
        </tbody>
      </table>
      {#snippet more()}
        {#if psiChecks.length > 0}
          <table>
            <thead><tr><th>ψ 群</th><th>ゲート</th><th>判定量</th><th>ε 以下</th></tr></thead>
            <tbody>
              {#each psiChecks as check (check.names.join(','))}
                {#each check.names as name, i (name)}
                  <tr>
                    <td>{i === 0 ? `ψ{${check.names.join(',')}}` : ''}</td>
                    <td>{name}</td>
                    <td>{check.consistency[i].toPrecision(3)}</td>
                    <td>{check.consistency[i] <= epsilon ? '○' : '×'}</td>
                  </tr>
                {/each}
              {/each}
            </tbody>
          </table>
        {:else}
          <p class="note">選んだ組み分けに、2 つ以上のゲートの ψ 群がない。</p>
        {/if}
      {/snippet}
      {#snippet info()}
        <p>φ 群に入れられるのは、感度が負で、静止電位で式(11)が小さい m だけ。h, n, a, c は感度が正、b は式(11)が 16 と大きい。ψ 群は感度の符号がそろうゲートどうし（h, n, a, c と m, b）で作る。s,r は Ca 系と一緒に元のまま残す。</p>
        {@html eq`\Big|\sum_{j\in\Psi_k}\alpha_jk_j-k_i\Big|\,\Big[\sum_{j\in\Psi_k}k_j\Big]^{-1}\ll1 \quad (i\in\Psi_k)`}
        <p>Kepler 式(10)。{@html tex`\alpha_j=D_j/\sum_{l\in\Psi_k}D_l`} は式(6)の重み。平衡点で {@html tex`\partial f_i/\partial v_i=-k_i`} と置いて導いた式なので、静止電位（{@html tex`I_0=0`} の安定な平衡点）でだけ評価する。そこでは全ての等価電位が {@html tex`V`} に一致する。ε は Kepler が定めていない目安。式(10)の列は、2 つ以上のゲートの ψ 群すべてでの左辺の最大で、ψ 群がなければ —。</p>
        <p>r は v ≤ −60、c は v &gt; −10 mV で定常値が平らで、そこでは重みが 0 になる。</p>
      {/snippet}
    </Card>

    <Card kicker="時間" title="入力" subtitle="元モデルと縮約に共通の注入電流密度">
      {#if result}
        <LineChart traces={[{ label: '入力', points: result.full.map(p => ({ x: p.time, y: input(p.time, result!.config) })) }]} xDomain={domain} xLabel="時間 (ms)" yDomain={[-20, 40]} yLabel="入力 (µA/cm²)" />
      {/if}
      {#snippet info()}
        <p>検証用に定義した準周期波形。原著図8の数値や刺激の再現ではない。A = 0 で一定電流、I₀ = A = 0 で無入力になる。</p>
        {@html eq`I_{\mathrm{ext}}(t)=I_0+\frac{A}{1.5}\left[\sin\left(\frac{2\pi t}{80}\right)+\frac12\sin\left(\frac{2\pi\sqrt2\,t}{80}\right)\right]`}
      {/snippet}
    </Card>

    <Card kicker="時間" title="見る組の軌道" subtitle={groupCheck ? verdictSummary(groupCheck.durations) : '元モデルの軌道の上で、ゲートの値・等価電位・重みを見る'}>
      {#snippet controls()}
        <Toggles label="見る変数" bind:selected={shown} options={variableOptions} />
      {/snippet}
      {#if result && shown.length > 0}
        <figure>
          <figcaption>図1 ゲートの値</figcaption>
          {#if gates.length > 0}
            <LineChart traces={gateTraces(result.full, gates)} xDomain={domain} xLabel="時間 (ms)" yDomain={[0, 1]} yLabel="ゲート値" />
          {:else}
            <p class="note">ゲートを選ぶとここに出る。</p>
          {/if}
        </figure>
        <figure>
          <figcaption>図2 等価電位</figcaption>
          <LineChart traces={equivalentTraces(result.full, gates, withVoltage)} xDomain={domain} xLabel="時間 (ms)" yDomain={[-100, 60]} yLabel="電位 (mV)" />
        </figure>
        <figure>
          <figcaption>{withVoltage ? '感度 ∂F/∂v（φ 群として）' : '重み α（ψ 群として）'}</figcaption>
          {#if groupCheck}
            {#if withVoltage}
              <LineChart traces={groupTraces(result.full, gates, true)} xDomain={domain} xLabel="時間 (ms)" yLabel="∂F/∂v (mS/cm²)" guides={[zeroLine]} />
            {:else}
              <LineChart traces={groupTraces(result.full, gates, false)} xDomain={domain} xLabel="時間 (ms)" yDomain={[-0.5, 1.5]} yLabel="重み α" guides={[zeroLine, oneLine]} />
            {/if}
            <p class="note">{firstFailures(result.full, groupCheck.checks) || '入力の 500 ms を通して、条件 1,2 を満たす。'}</p>
          {:else}
            <p class="note">{withVoltage ? 'V と一緒にゲートを 1 つ以上' : 'ゲートを 2 つ以上'}選ぶと判定する。</p>
          {/if}
        </figure>
      {:else if result}
        <p class="note">変数を選ぶとここに出る。</p>
      {/if}
      {#snippet info()}
        <p>図1 と図2 は原著の図1・図2 に当たる。等価電位は、電位を固定して待てばそのゲート値になる電位 {@html tex`v_x=\bar{x}^{-1}(x)`}。すべて mV になり、{@html tex`V`} と同じ目盛りで重ねられる。図2 でよく重なる変数どうしが、一つにまとめる候補。同じ変数は図の間で同じ色。</p>
        <p>{@html tex`\bar{c}`} は −10 mV より上で 1 に張り付くので、{@html tex`v_c`} は −10 mV で頭打ちになる。{@html tex`\bar{x}`} の値域の外に出た点は描かない。</p>
        <p>三つ目の図は、51.01-apply-kam の条件 1（等価電位に変換できる）と条件 2（重みが正）を、元モデルが実際にたどる各時刻の状態で確かめる。縮約後の重みは各時刻の状態で決まるので、平衡点だけでは足りない。V を含めれば膜電位側の群 φ、含めなければ回復側の群 ψ として見る。</p>
        <p>条件 1：群のすべてのゲートが等価電位 {@html tex`v_x=\bar{x}^{-1}(x)`} を持つこと。条件 2：感度 {@html tex`D_x=(\partial I/\partial x)\,\bar{x}'(v_x)`} の符号が、{withVoltage ? 'すべて負であること（φ 群）' : '群の中でそろうこと（ψ 群）。そのとき重み α はすべて 0 と 1 の間に入る'}。</p>
        <p>{@html tex`D_x`} は今のゲート値・{@html tex`V`}・{@html tex`\chi`} と、{@html tex`v_x`} での傾き {@html tex`\bar{x}'`} で計算する。等価電位がない時刻は曲線を描かない。ψ 群の速さ（式(10)）は一番上の図、φ 群の速さ（式(11)）は 51.01-apply-kam で見る。</p>
      {/snippet}
    </Card>

    <Card kicker="時間" title={result ? `縮約の比較：${groupingLabel(result.grouping)}` : '縮約の比較'} moreLabel="Ca 状態 χ と AHP ゲート q も比べる" subtitle={result ? `発火 ${spikes(result.full).length} 回／${spikes(result.reduced).length} 回、電位 RMSE ${difference('v').toFixed(2)} mV（原著 図8）` : undefined}>
      {#if result}
        <LineChart traces={traces('v')} xDomain={domain} xLabel="時間 (ms)" yDomain={[-80, 60]} yLabel="膜電位 (mV)" />
      {/if}
      {#snippet more()}
        {#if result}
          <figure>
            <figcaption>Ca 状態 χ <small>RMSE {difference('chi').toFixed(3)}</small></figcaption>
            <LineChart traces={traces('chi')} xDomain={domain} xLabel="時間 (ms)" yLabel="χ（実装単位）" />
          </figure>
          <figure>
            <figcaption>AHP ゲート q <small>RMSE {difference('q').toFixed(5)}</small></figcaption>
            <LineChart traces={traces('q')} xDomain={domain} xLabel="時間 (ms)" yDomain={[0, 1]} yLabel="q" />
          </figure>
        {/if}
      {/snippet}
      {#snippet info()}
        <p>一番上の「群分けと式(10) の判定」で選んだ組を、51.01-apply-kam「縮約モデルの一般形」で積分する。φ 群のゲートは {@html tex`\bar{x}(\phi)`} に置き、{@html tex`\alpha_0`} は「φ 群の重み α₀」で、1 と置く（φ 群は速いゲートだけとみなす近似）か、各時刻に Kepler 式(9)の根を求めるかを選ぶ。ψ 群は各時刻の状態から式(6)の重みを計算する。発火数が一致しても、時刻や波形が一致するとは限らない。</p>
        {#if result}
          <p>
            発火は 0 mV の上向き通過で数え、標本間を線形補間する。元モデル (ms)：{spikes(result.full)
              .map(t => t.toFixed(2))
              .join(', ') || 'なし'}
          </p>
          <p>
            縮約 (ms)：{spikes(result.reduced)
              .map(t => t.toFixed(2))
              .join(', ') || 'なし'}
          </p>
        {/if}
        <p>初期条件：V = −60 mV、電位依存ゲートはそこでの定常値、すべての ψ = −60 mV。χ は Ca 電流と減衰の釣り合いから、q はその χ での定常値から初期化する。 各モデルを無入力で 200 ms 動かした後を t = 0 とし、刺激を 500 ms 与える。</p>
        <p>積分：ODEX の適応刻み（最大刻み 0.1 ms、絶対許容誤差は相対許容誤差の 0.1 倍）、表示標本は 0.1 ms 間隔。重みが正に定義できない場合は停止し、クリップはしない。</p>
        <p>入力が負になると膜電位が {@html tex`E_K`} を下回り、K の電流に入るゲートの感度の符号が反転して、ψ 群の重みが正に定まらなくなる（I₀ = 2、A = 5 など）。そのときは停止する。</p>
        <p><a href="https://doi.org/10.1007/BF00197717">Kepler et al. (1992) 図8</a>は HH + A電流モデルの 6 状態と 3 状態の比較で、この図の Traub とは対象が異なる。</p>
      {/snippet}
    </Card>
  {/snippet}
</Dashboard>
