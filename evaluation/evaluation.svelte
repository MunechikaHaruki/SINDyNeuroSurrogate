<script lang="ts">
  import { onMount } from 'svelte';
  import Dashboard from '$lib/Dashboard.svelte';
  import Card from '$lib/Card.svelte';
  import Range from '$lib/Range.svelte';
  import Choice from '$lib/Choice.svelte';
  import Toggles from '$lib/Toggles.svelte';
  import LineChart from '$lib/LineChart.svelte';
  import { RUNS, decimate, experimentColumns, gatesOf, listRuns, readComp, readCurrent, readInternal, readJson, readStatus, seriesEntry, type Column, type Metrics, type PairMetrics, type Point, type SeriesEntry, type WaveMeta } from './read/runs.ts';
  import type { IndexEntry, Status } from './orchestrate/spec.ts';

  type Trace = { label: string; points: Point[]; dashed?: boolean };
  /** 表示用に間引くときの区間の数。1 本あたり最大でこの 2 倍の点になる。 */
  const BUCKETS = 400;

  // ---- runs/ から読んだもの ----
  let index = $state<IndexEntry[]>([]);
  let status = $state<Status>({ experiments: [], jobs: [] });
  let columns = $state<Column[]>([]);
  let originalMeta = $state<WaveMeta>();
  let metas = $state<Record<string, WaveMeta>>({});
  let metrics = $state<Record<string, Metrics>>({});
  let summaries = $state<Record<string, Record<string, number | null>>>({});
  let error = $state('');

  // ---- 選ぶもの ----
  let experimentName = $state('');
  let seriesName = $state('');
  let chosen = $state<string[]>([]);
  let evalComp = $state('soma');
  let viewComps = $state<string[]>(['soma']);
  let metricKey = $state('spike_count');
  let yAuto = $state(true);
  let yMin = $state(0);
  let yMax = $state(1);
  let point = $state(0);
  let detailRun = $state('');
  let spikeOriginal = $state(0);
  let spikeSurrogate = $state(0);
  let windowStart = $state(0);
  let windowLength = $state(1000);

  let experiment = $derived(status.experiments.find(e => e.name === experimentName) ?? status.experiments[0]);
  /** 実験の系列のうち、原系を回したもの。 */
  let series = $derived.by(() => {
    const list: SeriesEntry[] = [];
    for (const declared of experiment?.series ?? []) {
      const found = experiment ? seriesEntry(index, experiment, declared.series) : undefined;
      if (found) list.push(found);
    }
    return list;
  });
  let entry = $derived(series.find(s => s.name === seriesName) ?? series[0]);
  // 名前を選んでいなければ最初の実験になるので、選択欄にもその名前を出す。
  $effect(() => {
    if (experiment && experimentName !== experiment.name) experimentName = experiment.name;
  });
  $effect(() => {
    if (entry && seriesName !== entry.name) seriesName = entry.name;
  });

  /** 選んだ列を、run の一覧の並びのまま（凡例の並び）。 */
  let selected = $derived(columns.filter(c => chosen.includes(c.train)));
  /** 凡例と表の見出しに使う短い名前。この系列の全列の名前に共通する語を省く（選び方で名前が変わらない）。 */
  let shortNames = $derived.by(() => {
    const names: Record<string, string> = {};
    for (const column of columns) {
      const kept: string[] = [];
      for (const word of column.trainName.split(' ')) {
        let shared = columns.length > 1;
        for (const other of columns) if (!other.trainName.split(' ').includes(word)) shared = false;
        if (!shared) kept.push(word);
      }
      names[column.train] = kept.length > 0 ? kept.join(' ') : column.trainName;
    }
    return names;
  });
  let detail = $derived(selected.find(c => c.train === detailRun) ?? selected[0]);
  let compNames = $derived(originalMeta?.names ? Object.keys(originalMeta.names) : []);
  let compId = $derived(originalMeta?.names?.[evalComp] ?? 0);
  let duration = $derived(originalMeta ? originalMeta.n[point] * originalMeta.dt : 0);
  let domain = $derived<[number, number]>([windowStart, Math.min(duration, windowStart + windowLength)]);

  /** run の一覧と実験の状態を読む。操作盤から ?experiment=…&series=… で来たらその系列を選ぶ。 */
  async function load(): Promise<void> {
    index = await listRuns();
    status = await readStatus();
    const params = new URLSearchParams(window.location.search);
    experimentName = params.get('experiment') ?? '';
    seriesName = params.get('series') ?? '';
  }

  onMount(() => {
    load().catch(e => (error = `runs/（${RUNS}）を読めない：${e instanceof Error ? e.message : String(e)}`));
  });

  // 系列を選んだら、その系列の列と、全部の列の目録・指標・要約を読む。
  $effect(() => {
    const current = entry;
    const chosenExperiment = experiment;
    if (!current || !chosenExperiment) return;
    void (async () => {
      const found = experimentColumns(chosenExperiment, current.name);
      const meta = await readJson<WaveMeta>(current.run, 'waves/meta.json');
      for (const column of found) {
        metas[column.run] = await readJson<WaveMeta>(column.run, 'waves/meta.json');
        metrics[column.run] = await readJson<Metrics>(column.run, 'metrics.json');
        summaries[column.run] = await readJson<Record<string, number | null>>(column.train, 'summary.json');
      }
      originalMeta = meta;
      columns = found;
      chosen = found.map(c => c.train);
      detailRun = found[0]?.train ?? '';
      point = 0;
      windowStart = 0;
      windowLength = meta.n[0] * meta.dt;
      if (!meta.names || !(evalComp in meta.names)) evalComp = Object.keys(meta.names ?? { soma: 0 })[0];
    })().catch(e => (error = e instanceof Error ? e.message : String(e)));
  });

  // ---- 波形：点ごとに、原系と選んだ列の電位を読む ----
  type Waves = { original: Record<string, Float32Array>; columns: Record<string, Record<string, Float32Array>> };
  let grid = $state<Waves[]>([]);
  $effect(() => {
    const meta = originalMeta;
    const shown = selected;
    const id = compId;
    if (!meta || !entry) return;
    const originalRun = entry.run;
    void (async () => {
      const list: Waves[] = [];
      for (let k = 0; k < meta.n.length; k++) {
        const waves: Waves = { original: await readComp(originalRun, 'waves', meta, k, id), columns: {} };
        for (const column of shown) waves.columns[column.run] = await readComp(column.run, 'waves', metas[column.run], k, id);
        list.push(waves);
      }
      grid = list;
    })().catch(e => (error = e instanceof Error ? e.message : String(e)));
  });

  /** 点の見出し（掃引値と単位）。単発なら空。 */
  function pointLabel(index: number): string {
    if (!entry || entry.param === null) return '';
    return `${entry.param} = ${entry.values[index]} ${entry.unit}`;
  }
  /** 1 点の電位：原系（実線）と各列（破線）。発散した列は描かない。 */
  function voltageTraces(waves: Waves, index: number): Trace[] {
    if (!originalMeta) return [];
    const list: Trace[] = [{ label: '原系', points: decimate(waves.original.V, originalMeta.t0, originalMeta.dt, domain[0], domain[1], BUCKETS) }];
    for (const column of selected) {
      const values = waves.columns[column.run];
      if (!values || diverged(column, index)) continue;
      list.push({ label: shortNames[column.train], points: decimate(values.V, originalMeta.t0, originalMeta.dt, domain[0], domain[1], BUCKETS), dashed: true });
    }
    return list;
  }
  function pair(column: Column, index: number): PairMetrics | undefined {
    return metrics[column.run]?.[evalComp]?.[index];
  }
  function diverged(column: Column, index: number): boolean {
    return pair(column, index)?.diverged ?? false;
  }
  /** 発散した列の名前（その点で）。 */
  function divergedNames(index: number): string {
    const names: string[] = [];
    for (const column of selected) if (diverged(column, index)) names.push(shortNames[column.train]);
    return names.join('、');
  }

  // ---- 指標の折れ線：掃引値に沿って ----
  let metricKeys = $derived.by(() => {
    const first = selected[0] ? metrics[selected[0].run]?.[evalComp]?.[0] : undefined;
    if (!first) return [];
    const keys = Object.keys(first.rows);
    for (const key of Object.keys(first.scalars)) keys.push(key);
    return keys;
  });
  /** 1 点・1 列の指標の値。rows の指標は [原系, 置換系]、scalars は置換系だけ。 */
  function metricValue(column: Column, index: number, side: 'original' | 'surrogate'): number | null {
    const found = pair(column, index);
    if (!found) return null;
    if (metricKey in found.rows) return found.rows[metricKey][side === 'original' ? 0 : 1];
    return side === 'surrogate' ? (found.scalars[metricKey] ?? null) : null;
  }
  function metricTraces(): Trace[] {
    if (!entry || selected.length === 0) return [];
    const list: Trace[] = [];
    if (metricKey in (pair(selected[0], 0)?.rows ?? {})) {
      const points: Point[] = [];
      for (let k = 0; k < entry.values.length; k++) {
        const value = metricValue(selected[0], k, 'original');
        if (value !== null) points.push({ x: entry.values[k], y: value });
      }
      list.push({ label: '原系', points });
    }
    for (const column of selected) {
      const points: Point[] = [];
      for (let k = 0; k < entry.values.length; k++) {
        const value = metricValue(column, k, 'surrogate');
        if (value !== null) points.push({ x: entry.values[k], y: value });
      }
      list.push({ label: shortNames[column.train], points, dashed: true });
    }
    return list;
  }
  let sweepDomain = $derived<[number, number]>(entry && entry.values.length > 1 ? [Math.min(...entry.values), Math.max(...entry.values)] : [0, 1]);

  // ---- 1 ペアの詳細 ----
  let detailWaves = $state<{ original: Record<string, Float32Array>; surrogate: Record<string, Float32Array>; projected?: Record<string, Float32Array>; current: Float32Array; comps: Record<string, Record<string, Float32Array>>; internal?: Record<string, Float32Array> }>();
  $effect(() => {
    const meta = originalMeta;
    const column = detail;
    const k = point;
    const id = compId;
    const comp = evalComp;
    const others = viewComps;
    if (!meta || !entry || !column) return;
    const originalRun = entry.run;
    void (async () => {
      const projectedRoot = `projected/${comp}`;
      const projected = entry!.replaceTargets.includes(comp) ? await readComp(column.run, projectedRoot, await readJson<WaveMeta>(column.run, `${projectedRoot}/meta.json`), k, id) : undefined;
      const comps: Record<string, Record<string, Float32Array>> = {};
      for (const name of others) comps[name] = await readComp(originalRun, 'waves', meta, k, meta.names![name]);
      const internal = others.length > 0 && meta.nodes ? await readInternal(originalRun, 'waves', meta, k) : undefined;
      detailWaves = {
        original: await readComp(originalRun, 'waves', meta, k, id),
        surrogate: await readComp(column.run, 'waves', metas[column.run], k, id),
        projected,
        current: await readCurrent(originalRun, 'waves', k),
        comps,
        internal,
      };
    })().catch(e => (error = e instanceof Error ? e.message : String(e)));
  });
  /** 置換系の潜在変数 z1, z2, …（射影した原系と重ねる）。 */
  let latents = $derived(detailWaves?.projected ? Object.keys(detailWaves.projected).filter(name => name !== 'V') : []);
  function overlay(original: Float32Array, surrogate: Float32Array | undefined): Trace[] {
    const t0 = originalMeta!.t0;
    const dt = originalMeta!.dt;
    const list: Trace[] = [{ label: '原系', points: decimate(original, t0, dt, domain[0], domain[1], BUCKETS) }];
    if (surrogate && detail && !diverged(detail, point)) list.push({ label: '置換系', points: decimate(surrogate, t0, dt, domain[0], domain[1], BUCKETS), dashed: true });
    return list;
  }
  /** 相平面の 1 本あたりの点の上限。1 本の折れ線は 1 つの path なので、時間の図より多く描ける。 */
  const PHASE_POINTS = 10000;
  /** 相平面（V と第 1 潜在変数）。時刻順に結ぶので、間引きは等間隔に取る。 */
  function phase(v: Float32Array, z: Float32Array): Point[] {
    const points: Point[] = [];
    const first = Math.max(0, Math.ceil((domain[0] - originalMeta!.t0) / originalMeta!.dt));
    const last = Math.min(v.length - 1, Math.floor((domain[1] - originalMeta!.t0) / originalMeta!.dt));
    const stride = Math.max(1, Math.floor((last - first) / PHASE_POINTS));
    for (let i = first; i <= last; i += stride) if (Number.isFinite(v[i]) && Number.isFinite(z[i])) points.push({ x: v[i], y: z[i] });
    return points;
  }
  function phaseTraces(): Trace[] {
    if (!detailWaves?.projected || latents.length === 0) return [];
    const list: Trace[] = [{ label: '原系（射影）', points: phase(detailWaves.projected.V, detailWaves.projected[latents[0]]) }];
    if (detail && !diverged(detail, point)) list.push({ label: '置換系', points: phase(detailWaves.surrogate.V, detailWaves.surrogate[latents[0]]), dashed: true });
    return list;
  }
  /** 全系列の横軸の範囲。 */
  function extent(traces: Trace[]): [number, number] {
    let low = Infinity;
    let high = -Infinity;
    for (const trace of traces) {
      for (const p of trace.points) {
        low = Math.min(low, p.x);
        high = Math.max(high, p.x);
      }
    }
    return low < high ? [low, high] : [low - 1, low + 1];
  }
  /** 時間の図の 1 本（表示幅で間引く）。 */
  function timeTrace(label: string, values: Float32Array): Trace {
    return { label, points: decimate(values, originalMeta!.t0, originalMeta!.dt, domain[0], domain[1], BUCKETS) };
  }
  /** 全区画の図に出す区画の電位。 */
  function viewVoltages(): Trace[] {
    const list: Trace[] = [];
    for (const name of viewComps) {
      const values = detailWaves?.comps[name];
      if (values) list.push(timeTrace(name, values.V));
    }
    return list;
  }
  /** 全区画の図に出す区画の内部電流（結合電流と、刺激した区画では注入電流の和）。 */
  function viewInternal(): Trace[] {
    const list: Trace[] = [];
    for (const name of viewComps) {
      const values = detailWaves?.internal?.[String(originalMeta!.names![name])];
      if (values) list.push(timeTrace(name, values));
    }
    return list;
  }
  /** 全区画の図に出す区画のゲート。区画が 1 つなら変数名だけを凡例にする。 */
  function viewGates(): Trace[] {
    const list: Trace[] = [];
    for (const name of viewComps) {
      const values = detailWaves?.comps[name];
      if (!values) continue;
      for (const gate of gatesOf(originalMeta!, originalMeta!.names![name])) {
        const label = viewComps.length > 1 ? `${gate}（${name}）` : gate;
        list.push(timeTrace(label, values[gate]));
      }
    }
    return list;
  }
  let detailPair = $derived(detail ? pair(detail, point) : undefined);
  function format(value: number | null | undefined): string {
    if (value === null || value === undefined) return '—';
    if (Number.isInteger(value)) return String(value);
    return Math.abs(value) >= 1000 || (value !== 0 && Math.abs(value) < 0.01) ? value.toExponential(2) : value.toFixed(3);
  }
  function difference(o: number | null, s: number | null): number | null {
    return o === null || s === null ? null : o - s;
  }
  let summaryKeys = $derived.by(() => {
    const keys: string[] = [];
    for (const column of selected) for (const key of Object.keys(summaries[column.run] ?? {})) if (!keys.includes(key)) keys.push(key);
    return keys;
  });
</script>

<Dashboard title="SINDy サロゲート：評価の比較" summary="学習したサロゲートで Traub の区画を置き換え、同じ入力の掃引で原系と比べる。点ごとの膜電位、掃引値に沿った指標、1 組の詳細（潜在変数・相平面・スパイクの特徴量）と、学習側の指標を並べる。">
  {#snippet sidebar()}
    {#if error}<p class="note" aria-live="polite">停止：{error}</p>{/if}
    <section>
      <h3>実験</h3>
      <Choice label="見比べる組" bind:value={experimentName} options={status.experiments.map(e => ({ value: e.name, label: e.name }))} />
    </section>
    <section>
      <h3>系列</h3>
      {#if series.length > 0}
        <Choice label="評価した系列" bind:value={seriesName} options={series.map(s => ({ value: s.name, label: s.name }))} />
        {#if entry}<p class="note">置換：{entry.replaceTargets.join(', ')}。{entry.param ? `${entry.param} を ${entry.values.join(', ')} ${entry.unit} で振る。` : '単発。'}</p>{/if}
      {:else if !error}
        <p class="note">この実験には評価した系列が無い。<a href="runs">別冊「SINDy サロゲート：run の操作盤」</a> から回す。</p>
      {/if}
    </section>
    <section>
      <h3>比べる学習 run</h3>
      <Toggles label="置換した列" bind:selected={chosen} options={columns.map(c => ({ value: c.train, label: c.trainName }))} />
      <p class="note">宣言にあってまだ回していない組は <a href="runs">別冊「SINDy サロゲート：run の操作盤」</a> から回す。</p>
    </section>
    <section>
      <h3>見る区画</h3>
      <Choice label="比べる区画" bind:value={evalComp} options={compNames.map(name => ({ value: name, label: name }))} />
      <Toggles label="全区画の図に出す区画" bind:selected={viewComps} options={compNames.map(name => ({ value: name, label: name }))} />
    </section>
    <section>
      <h3>時間の表示</h3>
      <Range label="表示開始" bind:value={windowStart} min={0} max={Math.max(0, duration - 10)} step={10} display={`${windowStart} ms`} />
      <Range label="表示幅" bind:value={windowLength} min={10} max={Math.max(10, duration)} step={10} display={`${Math.min(windowLength, duration - windowStart).toFixed(0)} ms`} />
    </section>
  {/snippet}

  {#snippet left()}
    <Card kicker="点ごと" title="膜電位" subtitle={`${evalComp} の電位。原系（実線）と置換系（破線）`}>
      {#if originalMeta && grid.length === originalMeta.n.length}
        <figure>
          {#each grid as waves, k (k)}
            <figure>
              <figcaption>
                {pointLabel(k) || '単発'}
                {#if divergedNames(k)}<small>発散：{divergedNames(k)}</small>{/if}
              </figcaption>
              <LineChart traces={voltageTraces(waves, k)} xDomain={domain} xLabel="時間 (ms)" yLabel="膜電位 (mV)" />
            </figure>
          {/each}
        </figure>
      {:else if entry}
        <p class="note">波形を読んでいる。</p>
      {/if}
      {#snippet info()}
        <p>波形は表示幅を {BUCKETS} 区間に分け、区間ごとの最小と最大だけを描く（スパイクの山と谷は残る）。置換系の電位が NaN・無限大か ±1000 mV を超えた列は、その点では描かない。</p>
      {/snippet}
    </Card>

    {#if entry && entry.values.length > 1}
      <Card kicker="掃引" title="指標と掃引値" subtitle={`${evalComp} の ${metricKey}`}>
        {#snippet controls()}
          <Choice label="指標" bind:value={metricKey} options={metricKeys.map(key => ({ value: key, label: key }))} />
          <label><input type="checkbox" bind:checked={yAuto} /> 縦軸を自動で決める</label>
          {#if !yAuto}
            <label>下限 <input type="number" bind:value={yMin} step="any" /></label>
            <label>上限 <input type="number" bind:value={yMax} step="any" /></label>
          {/if}
        {/snippet}
        <LineChart traces={metricTraces()} xDomain={sweepDomain} yDomain={yAuto ? undefined : [yMin, yMax]} xLabel={`${entry.param} (${entry.unit})`} yLabel={metricKey} />
        {#snippet info()}
          <p>spike_count・latency・mean_isi・std_isi は原系（実線）と置換系（破線）の両方を、rmse・mae・periodicity_gap・spike_shape_corr は原系と比べた値なので置換系だけを描く。スパイクと ISI は eFEL で数える。</p>
        {/snippet}
      </Card>
    {/if}

    <Card kicker="1 組" title={detail ? `詳細：${detail.trainName}` : '詳細'} subtitle={pointLabel(point) || '単発'} moreLabel="全区画の波形（原系）">
      {#snippet controls()}
        <Choice label="見る列" bind:value={detailRun} options={selected.map(c => ({ value: c.train, label: c.trainName }))} />
        {#if originalMeta && originalMeta.n.length > 1}
          <Range label="点" bind:value={point} min={0} max={originalMeta.n.length - 1} step={1} display={pointLabel(point)} />
        {/if}
      {/snippet}
      {#if detailWaves}
        <figure>
          <figcaption>膜電位 V <small>{evalComp}</small></figcaption>
          <LineChart traces={overlay(detailWaves.original.V, detailWaves.surrogate.V)} xDomain={domain} xLabel="時間 (ms)" yLabel="膜電位 (mV)" />
        </figure>
        {#if latents.length > 0}
          <figure>
            {#each latents as name (name)}
              <figure>
                <figcaption>潜在変数 {name} <small>原系は射影</small></figcaption>
                <LineChart traces={overlay(detailWaves.projected![name], detailWaves.surrogate[name])} xDomain={domain} xLabel="時間 (ms)" yLabel={name} />
              </figure>
            {/each}
          </figure>
        {/if}
        {#if latents.length > 0}
          {@const traces = phaseTraces()}
          <figure>
            <figcaption>相平面 <small>V と {latents[0]}</small></figcaption>
            <LineChart {traces} xDomain={extent(traces)} xLabel="膜電位 V (mV)" yLabel={latents[0]} />
          </figure>
        {:else}
          <p class="note">{evalComp} は置換していないので、潜在変数と相平面は無い。</p>
        {/if}
      {:else if detail}
        <p class="note">波形を読んでいる。</p>
      {/if}
      {#snippet more()}
        {#if detailWaves && viewComps.length > 0}
          <figure>
            <figcaption>膜電位 V</figcaption>
            <LineChart traces={viewVoltages()} xDomain={domain} xLabel="時間 (ms)" yLabel="膜電位 (mV)" />
          </figure>
          {#if detailWaves.internal}
            <figure>
              <figcaption>内部電流 I_internal <small>隣の区画との結合電流。刺激した区画は注入電流も含む</small></figcaption>
              <LineChart traces={viewInternal()} xDomain={domain} xLabel="時間 (ms)" yLabel="I_internal (µA/cm²)" />
            </figure>
          {/if}
          <figure>
            <figcaption>ゲート</figcaption>
            <LineChart traces={viewGates()} xDomain={domain} xLabel="時間 (ms)" yLabel="ゲート" />
          </figure>
        {:else}
          <p class="note">操作列の「全区画の図に出す区画」で選ぶ。</p>
        {/if}
      {/snippet}
      {#snippet info()}
        <p>原系の潜在変数は、原系のゲートをこの列の前処理（PCA などの符号化）で潜在空間へ写したもの。置換系の潜在変数と重なれば、置換した区画が同じ軌道をたどっている。</p>
      {/snippet}
    </Card>

    {#if detailPair}
      <Card kicker="1 組" title="指標" subtitle={`${evalComp}、${pointLabel(point) || '単発'}`}>
        {#snippet controls()}
          <Range label="原系のスパイク番号" bind:value={spikeOriginal} min={0} max={Math.max(0, (detailPair.spikes.peak_voltage?.[0].length ?? 1) - 1)} step={1} />
          <Range label="置換系のスパイク番号" bind:value={spikeSurrogate} min={0} max={Math.max(0, (detailPair.spikes.peak_voltage?.[1].length ?? 1) - 1)} step={1} />
        {/snippet}
        <table>
          <thead><tr><th>指標</th><th>原系</th><th>置換系</th><th>差</th></tr></thead>
          <tbody>
            {#each Object.entries(detailPair.rows) as [key, values] (key)}
              <tr><td>{key}</td><td>{format(values[0])}</td><td>{format(values[1])}</td><td>{format(difference(values[0], values[1]))}</td></tr>
            {/each}
            {#each Object.entries(detailPair.scalars) as [key, value] (key)}
              <tr><td>{key}</td><td>—</td><td>{format(value)}</td><td>—</td></tr>
            {/each}
          </tbody>
        </table>
        <table>
          <thead><tr><th>スパイクの特徴量</th><th>原系 #{spikeOriginal}</th><th>置換系 #{spikeSurrogate}</th><th>差</th></tr></thead>
          <tbody>
            {#each Object.entries(detailPair.spikes) as [key, values] (key)}
              {@const o = values[0][spikeOriginal] ?? null}
              {@const s = values[1][spikeSurrogate] ?? null}
              <tr><td>{key}</td><td>{format(o)}</td><td>{format(s)}</td><td>{format(difference(o, s))}</td></tr>
            {/each}
          </tbody>
        </table>
      </Card>
    {/if}
  {/snippet}

  {#snippet right()}
    <Card kicker="原系" title="入力" subtitle={pointLabel(point) || '単発'}>
      {#if detailWaves && originalMeta}
        <LineChart traces={[{ label: 'I_ext', points: decimate(detailWaves.current, originalMeta.t0, originalMeta.dt, domain[0], domain[1], BUCKETS) }]} xDomain={domain} xLabel="時間 (ms)" yLabel="I_ext (µA/cm²)" />
      {/if}
    </Card>

    <Card kicker="学習" title="学習側の指標" subtitle="疎さ、前処理の寄与率、演算コスト">
      {#if selected.length > 0}
        <table>
          <thead
            ><tr
              ><th>指標</th>{#each selected as column (column.train)}<th>{shortNames[column.train]}</th>{/each}</tr
            ></thead
          >
          <tbody>
            {#each summaryKeys as key (key)}
              <tr
                ><td>{key}</td>{#each selected as column (column.train)}<td>{format(summaries[column.run]?.[key])}</td>{/each}</tr
              >
            {/each}
          </tbody>
        </table>
      {/if}
      {#snippet info()}
        <p>cost/* は 1 区画 1 ステップあたりの演算回数（exp・div・加減 pm・mul）。cost/surr-orig が負なら置換系の方が少ない。</p>
      {/snippet}
    </Card>
  {/snippet}
</Dashboard>
