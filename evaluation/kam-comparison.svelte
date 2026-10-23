<script lang="ts">
  import { onMount } from 'svelte';
  import Dashboard from '$lib/book/Dashboard.svelte';
  import Card from '$lib/Card.svelte';
  import Range from '$lib/Range.svelte';
  import Choice from '$lib/Choice.svelte';
  import LineChart from '$lib/LineChart.svelte';
  import { RUNS, decimate, experimentColumns, listRuns, readComp, readJson, readStatus, seriesEntry, type Column, type Point, type SeriesEntry, type WaveMeta } from './read/runs.ts';
  import { crossingDiff, crossings, type CrossingDiff } from './read/compare.ts';
  import { gateNames, simulateCurrent, type GateName, type Grouping } from './traub/traub.ts';
  import type { IndexEntry, Status } from './orchestrate/spec.ts';

  type Trace = { label: string; points: Point[]; dashed?: boolean };
  /** 群ごとに圧縮した実験。どれも単一区画の Traub を SERIES で評価する。 */
  const EXPERIMENTS = ['grouped_hn', 'grouped_hn_ac'];
  const SERIES = 'traub_somastim';
  /** SERIES と同じ電流：静穏 10 ms のあと定常電流を 300 ms（conf.ts の STIM）。 */
  const SILENCE = 10;
  const LENGTH = 310;
  /** TS の縮約モデルの標本の間隔（traub.ts の sampleStep）。 */
  const TS_DT = 0.1;
  const BUCKETS = 400;

  /** 学習した列と、同じ群を KAM でまとめた縮約。 */
  type Pair = Column & { groups: string[][]; kam: Grouping };
  /** 表の 1 行：学習した列か、KAM の縮約。 */
  type Row = { label: string; diffs: CrossingDiff[] };

  let index = $state<IndexEntry[]>([]);
  let status = $state<Status>({ experiments: [], jobs: [] });
  let error = $state('');
  let entry = $state<SeriesEntry>();
  let meta = $state<WaveMeta>();
  let pairs = $state<Pair[]>([]);
  /** run の id → 点ごとの soma の電位。原系の run も入る。 */
  let python = $state<Record<string, Float64Array[]>>({});
  /** 縮約の鍵 → 点ごとの電位。鍵 'full' は TS の元のモデル。 */
  let reduced = $state<Record<string, Float64Array[]>>({});

  let absorb = $state(false);
  let detailRun = $state('');
  let metric = $state<'count' | 'peak' | 'lastShift'>('count');
  let windowStart = $state(0);
  let windowLength = $state(LENGTH);

  let detail = $derived(pairs.find(p => p.run === detailRun) ?? pairs[0]);
  let domain = $derived<[number, number]>([windowStart, Math.min(LENGTH, windowStart + windowLength)]);

  /** 学習した群を KAM の ψ 群にする。1 つだけの群は元の式なので、ψ 群にしない。 */
  function kamOf(groups: string[][], phi: GateName[]): Grouping {
    const psi: GateName[][] = [];
    for (const gates of groups) {
      if (gates.length < 2) continue;
      // ゲートは traub.ts の並び（m, h, n, a, …）にする
      const names: GateName[] = [];
      for (const name of gateNames) if (gates.includes(name.toUpperCase())) names.push(name);
      psi.push(names);
    }
    return { phi, psi };
  }
  function groupingKey(grouping: Grouping): string {
    return JSON.stringify(grouping);
  }
  /** 縮約を「KAM Φ{m}、ψ{h,n}、ψ{a,c}」の形に書く。 */
  function kamLabel(grouping: Grouping): string {
    const parts: string[] = [];
    if (grouping.phi.length > 0) parts.push(`Φ{${grouping.phi.join(',')}}`);
    for (const group of grouping.psi) parts.push(`ψ{${group.join(',')}}`);
    return `KAM ${parts.join('、')}`;
  }
  /** 表と図に出す縮約。m を吸収するかで φ 群が変わる。 */
  function kamFor(pair: Pair): Grouping {
    return absorb ? kamOf(pair.groups, ['m']) : pair.kam;
  }

  /** run の一覧と、比べる列を読む。 */
  async function load(): Promise<void> {
    index = await listRuns();
    status = await readStatus();
    const found: Pair[] = [];
    for (const name of EXPERIMENTS) {
      const experiment = status.experiments.find(e => e.name === name);
      if (!experiment) continue;
      entry ??= seriesEntry(index, experiment, SERIES);
      for (const column of experimentColumns(experiment, SERIES)) {
        const train = index.find(e => e.id === column.train);
        if (!train || train.spec.kind !== 'train') continue;
        const groups: string[][] = [];
        for (const group of train.spec.surrogate.groups) groups.push(group.gates);
        found.push({ ...column, groups, kam: kamOf(groups, []) });
      }
    }
    if (!entry) return;
    const original = await readJson<WaveMeta>(entry.run, 'waves/meta.json');
    python[entry.run] = await readVoltages(entry.run, original);
    for (const pair of found) python[pair.run] = await readVoltages(pair.run, await readJson<WaveMeta>(pair.run, 'waves/meta.json'));
    meta = original;
    pairs = found;
    detailRun = found[0]?.run ?? '';
  }
  async function readVoltages(run: string, waves: WaveMeta): Promise<Float64Array[]> {
    const list: Float64Array[] = [];
    for (let k = 0; k < waves.n.length; k++) list.push((await readComp(run, 'waves', waves, k, 0)).V);
    return list;
  }

  /** 縮約を全点で積分する。描画を止めないよう、1 本ごとに手を離す。 */
  async function integrate(key: string, grouping?: Grouping): Promise<void> {
    if (!entry || reduced[key]) return;
    const list: Float64Array[] = [];
    for (const value of entry.values) {
      await new Promise(resolve => setTimeout(resolve));
      const current = function (t: number): number {
        return t < SILENCE ? 0 : value;
      };
      const samples = simulateCurrent(current, LENGTH, grouping);
      const v = new Float64Array(samples.length);
      for (let i = 0; i < samples.length; i++) v[i] = samples[i].state[0];
      list.push(v);
    }
    reduced[key] = list;
  }

  onMount(() => {
    load().catch(e => (error = `runs/（${RUNS}）を読めない：${e instanceof Error ? e.message : String(e)}`));
  });
  // 列が揃うか、m の扱いを変えたら、まだ積分していない縮約を積分する。
  $effect(() => {
    const list = pairs;
    const phi = absorb;
    if (list.length === 0) return;
    void (async () => {
      await integrate('full');
      for (const pair of list) {
        const grouping = phi ? kamOf(pair.groups, ['m']) : pair.kam;
        await integrate(groupingKey(grouping), grouping);
      }
    })().catch(e => (error = e instanceof Error ? e.message : String(e)));
  });

  /** 学習した列は Python の原系と、KAM は TS の元のモデルと比べる（二つの実装は強い電流で発火が 1 回違う）。 */
  function pythonDiffs(run: string): CrossingDiff[] {
    const list: CrossingDiff[] = [];
    const original = entry ? python[entry.run] : undefined;
    const model = python[run];
    if (!original || !model || !meta) return list;
    for (let k = 0; k < original.length; k++) list.push(crossingDiff(crossings(original[k], meta.dt), crossings(model[k], meta.dt)));
    return list;
  }
  function kamDiffs(grouping: Grouping): CrossingDiff[] {
    const list: CrossingDiff[] = [];
    const full = reduced.full;
    const model = reduced[groupingKey(grouping)];
    if (!full || !model) return list;
    for (let k = 0; k < full.length; k++) list.push(crossingDiff(crossings(full[k], TS_DT), crossings(model[k], TS_DT)));
    return list;
  }
  /** 列ごとに学習した行を並べ、同じ群の KAM の行をその後ろに一度だけ置く。 */
  let rows = $derived.by(() => {
    const list: Row[] = [];
    const placed: string[] = [];
    for (const pair of pairs) {
      list.push({ label: pair.trainName, diffs: pythonDiffs(pair.run) });
      const grouping = kamFor(pair);
      const key = groupingKey(grouping);
      if (placed.includes(key)) continue;
      placed.push(key);
      list.push({ label: kamLabel(grouping), diffs: kamDiffs(grouping) });
    }
    return list;
  });
  function cell(diff: CrossingDiff | undefined): string {
    if (!diff) return '…';
    if (metric === 'count') return `${diff.count[0]} → ${diff.count[1]}`;
    const value = metric === 'peak' ? diff.peak : diff.lastShift;
    if (value === null) return '—';
    return metric === 'peak' && value > 0 ? `+${value.toFixed(1)}` : value.toFixed(1);
  }
  const metricOptions = [
    { value: 'count', label: '発火回数（元 → モデル）' },
    { value: 'peak', label: '山の高さの差（mV）' },
    { value: 'lastShift', label: '最後の山の時刻のずれ（ms）' },
  ];

  function trace(label: string, v: Float64Array, dt: number, dashed = false): Trace {
    return { label, points: decimate(v, 0, dt, domain[0], domain[1], BUCKETS), dashed };
  }
  /** 1 点の電位：Python の原系と学習した列、TS の元のモデルと KAM。 */
  function traces(k: number): Trace[] {
    const list: Trace[] = [];
    if (!entry || !meta || !detail) return list;
    list.push(trace('原系（Python）', python[entry.run][k], meta.dt));
    list.push(trace(detail.trainName, python[detail.run][k], meta.dt, true));
    const full = reduced.full?.[k];
    const kam = reduced[groupingKey(kamFor(detail))]?.[k];
    if (full) list.push(trace('原系（TS）', full, TS_DT));
    if (kam) list.push(trace(kamLabel(kamFor(detail)), kam, TS_DT, true));
    return list;
  }
</script>

<Dashboard title="群ごとの圧縮と KAM 縮約の比較" summary="Traub 単一区画で、群ごとに学習した圧縮と同じ群の KAM 縮約を定常電流で比べる。膜電位の重ね描きと、発火回数・山の高さ・時刻のずれの表。">
  {#snippet sidebar()}
    {#if error}<p class="note" aria-live="polite">停止：{error}</p>{/if}
    <section>
      <h3>比べるもの</h3>
      <p class="note">実験 {EXPERIMENTS.join('、')} を系列 {SERIES} で回した列と、その群を ψ 群にした KAM 縮約。学習した群は <a href="evaluation">別冊「SINDy サロゲート：評価の比較」</a>、KAM の作り方は <a href="group-comparison">別冊「Traub 単一区画：ゲートの組を選び、縮約と比べる」</a>。</p>
      <label><input type="checkbox" bind:checked={absorb} /> KAM で m を膜電位へ吸収する</label>
      {#if pairs.length === 0 && !error}<p class="note">回し終えた列が無い。<a href="runs">別冊「SINDy サロゲート：run の操作盤」</a> から回す。</p>{/if}
    </section>
    <section>
      <h3>時間の表示</h3>
      <Range label="表示開始" bind:value={windowStart} min={0} max={LENGTH - 10} step={10} display={`${windowStart} ms`} />
      <Range label="表示幅" bind:value={windowLength} min={10} max={LENGTH} step={10} display={`${Math.min(windowLength, LENGTH - windowStart)} ms`} />
    </section>
  {/snippet}

  {#snippet left()}
    <Card kicker="点ごと" title="膜電位" subtitle={detail ? `${detail.trainName} と ${kamLabel(kamFor(detail))}` : '列を読んでいる'}>
      {#snippet controls()}
        <Choice label="学習した列" bind:value={detailRun} options={pairs.map(p => ({ value: p.run, label: p.trainName }))} />
      {/snippet}
      {#if entry && meta && detail}
        <figure>
          {#each entry.values as value, k (k)}
            <figure>
              <figcaption>{entry.param} = {value} {entry.unit}</figcaption>
              <LineChart traces={traces(k)} xDomain={domain} xLabel="時間 (ms)" yLabel="膜電位 (mV)" />
            </figure>
          {/each}
        </figure>
      {/if}
      {#snippet info()}
        <p>学習した列は Python のシミュレータで、KAM は TS のシミュレータで積分する。二つの元のモデルは強い電流で発火が 1 回違うので、それぞれの元のモデル（実線）と重ねる。</p>
      {/snippet}
    </Card>
  {/snippet}

  {#snippet right()}
    <Card kicker="指標" title="元のモデルとの差" subtitle={SERIES}>
      {#snippet controls()}
        <Choice label="指標" bind:value={metric} options={metricOptions} />
      {/snippet}
      {#if entry}
        <table>
          <thead
            ><tr
              ><th>モデル</th>{#each entry.values as value (value)}<th>{value} {entry.unit}</th>{/each}</tr
            ></thead
          >
          <tbody>
            {#each rows as row (row.label)}
              <tr
                ><td>{row.label}</td>{#each entry.values as value, k (k)}<td title={String(value)}>{cell(row.diffs[k])}</td>{/each}</tr
              >
            {/each}
          </tbody>
        </table>
      {/if}
      {#snippet info()}
        <p>スパイクは 0 mV を上向きに越えた所で数え、山はそのスパイクの最大値。KAM にも同じ物差しを当てるため eFEL を使わない。時刻のずれは、両方にある最後のスパイクで測る。</p>
      {/snippet}
    </Card>
  {/snippet}
</Dashboard>
