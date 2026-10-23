<script lang="ts">
  import { onMount } from 'svelte';
  import Dashboard from '$lib/book/Dashboard.svelte';
  import Card from '$lib/Card.svelte';
  import Range from '$lib/Range.svelte';
  import Choice from '$lib/Choice.svelte';
  import LineChart from '$lib/LineChart.svelte';
  import { RUNS, decimate, experimentColumns, listRuns, readComp, readJson, readStatus, seriesEntry, type Column, type Point, type SeriesEntry, type WaveMeta } from './read/runs.ts';
  import { crossingDiff, crossings, type CrossingDiff } from './read/compare.ts';
  import type { IndexEntry } from './orchestrate/spec.ts';

  type Trace = { label: string; points: Point[]; dashed?: boolean };
  /** 群ごとに圧縮した実験。どれも単一区画の Traub を SERIES で評価し、同じ群の KAM 縮約（preset kam_*）を含む。 */
  const EXPERIMENTS = ['grouped_hn', 'grouped_hn_ac'];
  const SERIES = 'traub_somastim';
  const BUCKETS = 400;

  /** 回し終えた列と、その群（「{N,H}{A,C}」の形）と、KAM か。 */
  type Entry = Column & { groups: string; kam: boolean };

  let error = $state('');
  let entry = $state<SeriesEntry>();
  let meta = $state<WaveMeta>();
  let entries = $state<Entry[]>([]);
  /** run の id → 点ごとの soma の電位。原系の run も入る。 */
  let voltages = $state<Record<string, Float64Array[]>>({});

  let detailRun = $state('');
  let metric = $state<'count' | 'peak' | 'lastShift'>('count');
  let windowStart = $state(0);
  let windowLength = $state(0);

  let learned = $derived(entries.filter(e => !e.kam));
  let detail = $derived(learned.find(e => e.run === detailRun) ?? learned[0]);
  let duration = $derived(meta ? meta.n[0] * meta.dt : 0);
  let domain = $derived<[number, number]>([windowStart, Math.min(duration, windowStart + windowLength)]);

  /** 学習した列と同じ群の KAM の列。 */
  function kamFor(column: Entry): Entry | undefined {
    return entries.find(e => e.kam && e.groups === column.groups);
  }
  /** 群を「{N,H}{A,C}」の形に書く。 */
  function groupsOf(train: IndexEntry): string {
    let label = '';
    if (train.spec.kind !== 'train') return label;
    for (const group of train.spec.surrogate.groups) label += `{${group.gates.join(',')}}`;
    return label;
  }

  /** run の一覧と比べる列を読み、soma の電位を読む。 */
  async function load(): Promise<void> {
    const index = await listRuns();
    const status = await readStatus();
    const found: Entry[] = [];
    for (const name of EXPERIMENTS) {
      const experiment = status.experiments.find(e => e.name === name);
      if (!experiment) continue;
      entry ??= seriesEntry(index, experiment, SERIES);
      for (const column of experimentColumns(experiment, SERIES)) {
        const train = index.find(e => e.id === column.train);
        if (!train || train.spec.kind !== 'train') continue;
        found.push({ ...column, groups: groupsOf(train), kam: train.spec.surrogate.groups[0].compressor === 'kam' });
      }
    }
    if (!entry) return;
    const original = await readJson<WaveMeta>(entry.run, 'waves/meta.json');
    voltages[entry.run] = await readVoltages(entry.run, original);
    for (const column of found) voltages[column.run] = await readVoltages(column.run, await readJson<WaveMeta>(column.run, 'waves/meta.json'));
    meta = original;
    entries = found;
    detailRun = found.find(e => !e.kam)?.run ?? '';
    windowLength = original.n[0] * original.dt;
  }
  async function readVoltages(run: string, waves: WaveMeta): Promise<Float64Array[]> {
    const list: Float64Array[] = [];
    for (let k = 0; k < waves.n.length; k++) list.push((await readComp(run, 'waves', waves, k, 0)).V);
    return list;
  }

  onMount(() => {
    load().catch(e => (error = `runs/（${RUNS}）を読めない：${e instanceof Error ? e.message : String(e)}`));
  });

  /** 列の点ごとの、原系との差。どの列も同じ原系と比べる。 */
  function diffs(column: Entry): CrossingDiff[] {
    const list: CrossingDiff[] = [];
    const original = entry ? voltages[entry.run] : undefined;
    const model = voltages[column.run];
    if (!original || !model || !meta) return list;
    for (let k = 0; k < original.length; k++) list.push(crossingDiff(crossings(original[k], meta.dt), crossings(model[k], meta.dt)));
    return list;
  }
  /** 表の行：群ごとに学習した列を並べ、その後ろに同じ群の KAM を置く。 */
  let rows = $derived.by(() => {
    const groups: string[] = [];
    for (const column of entries) if (!groups.includes(column.groups)) groups.push(column.groups);
    const list: Entry[] = [];
    for (const group of groups) {
      for (const column of entries) if (column.groups === group && !column.kam) list.push(column);
      for (const column of entries) if (column.groups === group && column.kam) list.push(column);
    }
    return list;
  });
  function cell(diff: CrossingDiff): string {
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

  function trace(label: string, v: Float64Array, dashed = false): Trace {
    return { label, points: decimate(v, meta!.t0, meta!.dt, domain[0], domain[1], BUCKETS), dashed };
  }
  /** 1 点の電位：原系と、選んだ列と、同じ群の KAM。 */
  function traces(k: number): Trace[] {
    const list: Trace[] = [];
    if (!entry || !meta || !detail) return list;
    list.push(trace('原系', voltages[entry.run][k]));
    list.push(trace(detail.trainName, voltages[detail.run][k], true));
    const kam = kamFor(detail);
    if (kam) list.push(trace(kam.trainName, voltages[kam.run][k], true));
    return list;
  }
</script>

<Dashboard title="群ごとの圧縮と KAM 縮約の比較" summary="Traub 単一区画で、群ごとに学習した圧縮と同じ群の KAM 縮約を、同じシミュレータと原系で比べる。膜電位の重ね描きと、発火回数・山の高さ・時刻のずれの表。">
  {#snippet sidebar()}
    {#if error}<p class="note" aria-live="polite">停止：{error}</p>{/if}
    <section>
      <h3>比べるもの</h3>
      <p class="note">実験 {EXPERIMENTS.join('、')} を系列 {SERIES} で回した列。学習した圧縮と、同じ群を学習せずにまとめた KAM 縮約（ψ 群。m は元の式）を、同じシミュレータ・同じ原系で比べる。全指標は <a href="evaluation">別冊「SINDy サロゲート：評価の比較」</a>、KAM の作り方は <a href="group-comparison">別冊「Traub 単一区画：ゲートの組を選び、縮約と比べる」</a>。</p>
      {#if entries.length === 0 && !error}<p class="note">回し終えた列が無い。<a href="runs">別冊「SINDy サロゲート：run の操作盤」</a> から回す。</p>{/if}
    </section>
    <section>
      <h3>時間の表示</h3>
      <Range label="表示開始" bind:value={windowStart} min={0} max={Math.max(0, duration - 10)} step={10} display={`${windowStart} ms`} />
      <Range label="表示幅" bind:value={windowLength} min={10} max={Math.max(10, duration)} step={10} display={`${Math.min(windowLength, duration - windowStart).toFixed(0)} ms`} />
    </section>
  {/snippet}

  {#snippet left()}
    <Card kicker="点ごと" title="膜電位" subtitle={detail ? `${detail.trainName} と ${kamFor(detail)?.trainName ?? '同じ群の KAM（未実行）'}` : '列を読んでいる'}>
      {#snippet controls()}
        <Choice label="学習した列" bind:value={detailRun} options={learned.map(e => ({ value: e.run, label: e.trainName }))} />
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
        <p>どの列も Python の同じシミュレータ（前進 Euler、刻み 0.01 ms）で回し、同じ原系（実線）と重ねる。</p>
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
            {#each rows as row (row.run)}
              <tr
                ><td>{row.trainName}</td>{#each diffs(row) as diff, k (k)}<td>{cell(diff)}</td>{/each}</tr
              >
            {/each}
          </tbody>
        </table>
      {/if}
      {#snippet info()}
        <p>スパイクは 0 mV を上向きに越えた所で数え、山はそのスパイクの最大値。時刻のずれは、両方にある最後のスパイクで測る。</p>
      {/snippet}
    </Card>
  {/snippet}
</Dashboard>
