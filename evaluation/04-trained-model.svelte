<script lang="ts">
  import { onMount } from 'svelte';
  import Dashboard from '$lib/Dashboard.svelte';
  import Card from '$lib/Card.svelte';
  import Range from '$lib/Range.svelte';
  import Choice from '$lib/Choice.svelte';
  import Toggles from '$lib/Toggles.svelte';
  import LineChart from '$lib/LineChart.svelte';
  import SignedChart from '$lib/SignedChart.svelte';
  import MatrixTable from '$lib/MatrixTable.svelte';
  import NetChart from '$lib/NetChart.svelte';
  import { tex } from '$lib/tex.ts';
  import { RUNS, decimate, decodeComp, experimentTrains, readComp, readCurrent, readF32, readJson, readStatus, type Point, type WaveMeta } from './read/runs.ts';
  import type { Status } from './orchestrate/spec.ts';
  import { cumulative, featureTex, histogram, placeNet, targetTex, type ModelJson } from './read/model-json.ts';

  type Trace = { label: string; points: Point[]; dashed?: boolean };
  /** 学習 run の中身の置き場（neurosurrogate/run.py）。 */
  const ROOT = 'description';
  /** 時間の図を間引くときの区間の数。 */
  const BUCKETS = 400;
  /** 潜在の軌跡 1 本あたりの点の上限。 */
  const PATH_POINTS = 10000;
  /** 電位の分布の区間の数。 */
  const BINS = 60;

  /** 学習 comp 1 つ分：学習データの変数、同定器への入力、潜在、再構成 RMSE。 */
  type CompData = { vars: Record<string, Float32Array>; x: Record<string, Float32Array>; latent: Record<string, Float32Array>; recon: Float32Array };

  let status = $state<Status>({ experiments: [], jobs: [] });
  let experimentName = $state('');
  let runId = $state('');
  let model = $state<ModelJson>();
  let meta = $state<WaveMeta>();
  let comps = $state<Record<string, CompData>>({});
  let current = $state<Float32Array>();
  let error = $state('');

  let shownComps = $state<string[]>([]);
  let windowStart = $state(0);
  let windowLength = $state(1000);

  let duration = $derived(meta ? meta.n[0] * meta.dt : 0);
  let domain = $derived<[number, number]>([windowStart, Math.min(duration, windowStart + windowLength)]);
  /** 学習した区画のうち、表示するもの（学習した順）。 */
  let shown = $derived.by(() => {
    const names: string[] = [];
    for (const name of model?.net.train ?? []) if (shownComps.includes(name) && name in comps) names.push(name);
    return names;
  });

  let experiment = $derived(status.experiments.find(e => e.name === experimentName) ?? status.experiments[0]);
  // 名前を選んでいなければ最初の実験になるので、選択欄にもその名前を出す。
  $effect(() => {
    if (experiment && experimentName !== experiment.name) experimentName = experiment.name;
  });

  /** 実験の preset の、学習を終えた run（preset の順、掃引の点の順）。 */
  let runs = $derived(experiment ? experimentTrains(experiment) : []);

  /** 実験の状態を読み、操作盤から ?experiment=…&preset=… で来たらその preset の最初の run を選ぶ。 */
  async function load(): Promise<void> {
    status = await readStatus();
    const params = new URLSearchParams(window.location.search);
    experimentName = params.get('experiment') ?? '';
    const preset = experiment?.presets.find(p => p.preset === params.get('preset'));
    runId = preset?.runs[0]?.id ?? runs[0]?.id ?? '';
  }

  // 実験を選び直して、選んでいた run がその実験に無ければ、最初の run を選ぶ。
  $effect(() => {
    if (runs.length > 0 && !runs.some(r => r.id === runId)) runId = runs[0].id;
  });

  onMount(() => {
    load().catch(e => (error = `runs/（${RUNS}）を読めない：${e instanceof Error ? e.message : String(e)}`));
  });

  // run を選んだら、その中身を読む。学習 comp ごとの配列は最初に全部読む。
  $effect(() => {
    const id = runId;
    if (!id) return;
    model = undefined;
    meta = undefined;
    void (async () => {
      const readModel = await readJson<ModelJson>(id, `${ROOT}/model.json`);
      const readMeta = await readJson<WaveMeta>(id, `${ROOT}/train/meta.json`);
      const n = readMeta.n[0];
      /** 学習 comp 1 つ分を読む。区画どうしは並べて読む。 */
      async function load(name: string): Promise<CompData> {
        const compId = readMeta.names![name];
        const folder = `${ROOT}/fit/${compId}`;
        return {
          vars: await readComp(id, `${ROOT}/train`, readMeta, 0, compId),
          x: decodeComp((await readF32(id, `${folder}/x.f32`)).buffer as ArrayBuffer, readModel.x_names, n),
          latent: decodeComp((await readF32(id, `${folder}/latent.f32`)).buffer as ArrayBuffer, readModel.latent_names, n),
          recon: await readF32(id, `${folder}/recon_rmse.f32`),
        };
      }
      const readings: Promise<CompData>[] = [];
      for (const name of readModel.net.train) readings.push(load(name));
      const results = await Promise.all(readings);
      const loaded: Record<string, CompData> = {};
      for (let i = 0; i < results.length; i++) loaded[readModel.net.train[i]] = results[i];
      current = await readCurrent(id, `${ROOT}/train`, 0);
      comps = loaded;
      // 多区画で学習した run は全部重ねると読めないので、刺激の区画（学習していなければ先頭）から見せる。
      shownComps = readModel.net.train.includes(readModel.net.stim) ? [readModel.net.stim] : readModel.net.train.slice(0, 1);
      windowStart = 0;
      windowLength = n * readMeta.dt;
      meta = readMeta;
      model = readModel;
    })().catch(e => (error = e instanceof Error ? e.message : String(e)));
  });

  /** 時間の図の 1 本。 */
  function timeTrace(label: string, values: Float32Array): Trace {
    return { label, points: decimate(values, meta!.t0, meta!.dt, domain[0], domain[1], BUCKETS) };
  }
  /** 表示する区画ごとに 1 本ずつ。 */
  function perComp(pick: (data: CompData) => Float32Array): Trace[] {
    const list: Trace[] = [];
    for (const name of shown) list.push(timeTrace(name, pick(comps[name])));
    return list;
  }
  function voltageOf(data: CompData): Float32Array {
    return data.vars.V;
  }
  function reconOf(data: CompData): Float32Array {
    return data.recon;
  }
  /** 表示する先頭の区画の学習ゲート。 */
  function gateTraces(): Trace[] {
    const list: Trace[] = [];
    if (shown.length === 0 || !model) return list;
    for (const gate of model.train_gates) list.push(timeTrace(gate.toLowerCase(), comps[shown[0]].vars[gate]));
    return list;
  }
  /** 同定器への入力のうち、圧縮したもの（V は圧縮しないので出さない）。 */
  let inputNames = $derived.by(() => {
    const names: string[] = [];
    for (const name of model?.x_names ?? []) if (name !== 'V') names.push(name);
    return names;
  });
  function inputTraces(name: string): Trace[] {
    const list: Trace[] = [];
    for (const comp of shown) list.push(timeTrace(comp, comps[comp].x[name]));
    return list;
  }

  /** 潜在の軌跡：潜在が 2 次元以上なら z1 と z2、1 次元なら V と z1。時刻順に結ぶので、間引きは等間隔に取る。 */
  let pathAxes = $derived.by(() => {
    const names = model?.latent_names ?? [];
    if (names.length >= 2) return { x: names[0], y: names[1], xIsVoltage: false };
    return { x: 'V', y: names[0] ?? '', xIsVoltage: true };
  });
  function pathTraces(): Trace[] {
    const list: Trace[] = [];
    if (!meta) return list;
    const first = Math.max(0, Math.ceil((domain[0] - meta.t0) / meta.dt));
    for (const name of shown) {
      const data = comps[name];
      const xs = pathAxes.xIsVoltage ? data.vars.V : data.latent[pathAxes.x];
      const ys = data.latent[pathAxes.y];
      const last = Math.min(xs.length - 1, Math.floor((domain[1] - meta.t0) / meta.dt));
      const stride = Math.max(1, Math.floor((last - first) / PATH_POINTS));
      const points: Point[] = [];
      for (let i = first; i <= last; i += stride) points.push({ x: xs[i], y: ys[i] });
      list.push({ label: name, points });
    }
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

  /** 学習が踏んだ電位の分布（区画ごと）。区間は表示する区画の電位の最小から最大。 */
  function coverageTraces(): Trace[] {
    let low = Infinity;
    let high = -Infinity;
    for (const name of shown) {
      for (const v of comps[name].vars.V) {
        if (!Number.isFinite(v)) continue;
        low = Math.min(low, v);
        high = Math.max(high, v);
      }
    }
    const list: Trace[] = [];
    if (!(low < high)) return list;
    for (const name of shown) {
      const points: Point[] = [];
      for (const bin of histogram(comps[name].vars.V, low, high, BINS)) points.push({ x: bin.center, y: bin.count });
      list.push({ label: name, points });
    }
    return list;
  }

  // ---- 閉包項と前処理 ----
  let sindy = $derived(model?.closure.kind === 'sindy' ? (model.closure as { targets: string[]; features: string[]; xi: number[][] }) : undefined);
  let pca = $derived(model?.preprocessor.kind === 'pca' ? (model.preprocessor as { explained_variance_ratio: number[]; n_components: number }) : undefined);
  /** ξ を特徴量 × 対象変数に並べ替える（特徴量が多いので縦に流す）。 */
  function coefficientRows(): number[][] {
    const rows: number[][] = [];
    if (!sindy) return rows;
    for (let f = 0; f < sindy.features.length; f++) {
      const row: number[] = [];
      for (let t = 0; t < sindy.targets.length; t++) row.push(sindy.xi[t][f]);
      rows.push(row);
    }
    return rows;
  }
  function featureHeads(): string[] {
    const heads: string[] = [];
    for (const feature of sindy?.features ?? []) heads.push(tex`${featureTex(feature)}`);
    return heads;
  }
  function targetHeads(): string[] {
    const heads: string[] = [];
    for (const target of sindy?.targets ?? []) heads.push(tex`\dot{${targetTex(target)}}`);
    return heads;
  }
  /** 0 でない係数の数。 */
  let nonzero = $derived.by(() => {
    let count = 0;
    for (const row of sindy?.xi ?? []) for (const value of row) if (value !== 0) count += 1;
    return count;
  });
  function ratioPoints(): Point[] {
    const points: Point[] = [];
    const ratios = pca?.explained_variance_ratio ?? [];
    for (let i = 0; i < ratios.length; i++) points.push({ x: i + 1, y: ratios[i] });
    return points;
  }
  function cumulativeTrace(): Trace[] {
    const points: Point[] = [];
    const sums = cumulative(pca?.explained_variance_ratio ?? []);
    for (let i = 0; i < sums.length; i++) points.push({ x: i + 1, y: sums[i] });
    return [{ label: '累積寄与率', points }];
  }
  let kept = $derived(pca ? cumulative(pca.explained_variance_ratio)[pca.n_components - 1] : 0);

  // ---- ネット ----
  function netNodes(): { name: string; x: number; y: number; group: string; stim: boolean }[] {
    const list: { name: string; x: number; y: number; group: string; stim: boolean }[] = [];
    if (!model) return list;
    for (const node of placeNet(model.net)) {
      let group = 'その他の区画';
      if (model.net.replaceable.includes(node.name)) group = '置換できる区画';
      if (model.net.train.includes(node.name)) group = '学習した区画';
      list.push({ name: node.name, x: node.x, y: node.y, group, stim: node.name === model.net.stim });
    }
    return list;
  }
  function netEdges(): { from: string; to: string; label: string }[] {
    const list: { from: string; to: string; label: string }[] = [];
    for (const edge of model?.net.edges ?? []) list.push({ from: edge.src, to: edge.dst, label: edge.weight.toPrecision(2) });
    return list;
  }
</script>

<Dashboard title="SINDy サロゲート：学習 run の中身" summary="学習した 1 本の run について、閉包項の係数、前処理の寄与率、区画のつながりと、学習データから同定器へ渡したもの（圧縮した入力、再構成誤差、潜在の軌跡、電位の分布）を並べる。">
  {#snippet sidebar()}
    {#if error}<p class="note" aria-live="polite">停止：{error}</p>{/if}
    <section>
      <h3>実験</h3>
      <Choice label="見比べる組" bind:value={experimentName} options={status.experiments.map(e => ({ value: e.name, label: e.name }))} />
    </section>
    <section>
      <h3>学習 run</h3>
      {#if runs.length > 0}
        <Choice label="見る run" bind:value={runId} options={runs.map(r => ({ value: r.id, label: r.name }))} />
        <p class="note"><code>{runId}</code></p>
      {:else if !error}
        <p class="note">この実験には学習を終えた run が無い。</p>
      {/if}
      <p class="note">学習は <a href="03-runs">別冊「SINDy サロゲート：run の操作盤」</a> から回す。</p>
    </section>
    {#if model}
      <section>
        <h3>区画</h3>
        <Toggles label="表示する学習区画" bind:selected={shownComps} options={model.net.train.map(name => ({ value: name, label: name }))} />
      </section>
      <section>
        <h3>時間の表示</h3>
        <Range label="表示開始" bind:value={windowStart} min={0} max={Math.max(0, duration - 10)} step={10} display={`${windowStart} ms`} />
        <Range label="表示幅" bind:value={windowLength} min={10} max={Math.max(10, duration)} step={10} display={`${Math.min(windowLength, duration - windowStart).toFixed(0)} ms`} />
      </section>
    {/if}
  {/snippet}

  {#snippet left()}
    {#if model && meta && current}
      <Card kicker="ネット" title="区画のつながり" subtitle={`刺激 ${model.net.stim}、線の数字は結合の重み`}>
        <NetChart nodes={netNodes()} edges={netEdges()} />
      </Card>

      <Card kicker="学習データ" title="入力と応答" subtitle={`学習区画 ${shown.join('、') || 'なし'} の軌道`}>
        <figure>
          <figcaption>注入電流 <small>{model.net.stim}</small></figcaption>
          <LineChart traces={[timeTrace('I_ext', current)]} xDomain={domain} xLabel="時間 (ms)" yLabel="I_ext (µA/cm²)" />
        </figure>
        <figure>
          <figcaption>膜電位 V</figcaption>
          <LineChart traces={perComp(voltageOf)} xDomain={domain} xLabel="時間 (ms)" yLabel="膜電位 (mV)" />
        </figure>
        {#if shown.length > 0}
          <figure>
            <figcaption>学習したゲート <small>{shown[0]}</small></figcaption>
            <LineChart traces={gateTraces()} xDomain={domain} xLabel="時間 (ms)" yLabel="ゲート" />
          </figure>
        {/if}
        {#snippet info()}
          <p>学習データは保存しておらず、学習の設定（spec）からこの repo がシミュレーションし直したもの。ゲートは表示する先頭の区画だけを描く（ほかの区画のゲートも同じ多様体に乗る前提で、ずれは潜在の軌跡で見る）。</p>
        {/snippet}
      </Card>

      <Card kicker="学習データ" title="同定器への入力" subtitle="閉包項の同定に渡した直前の列（圧縮した後）">
        <figure>
          {#each inputNames as name (name)}
            <figure>
              <figcaption>{name}</figcaption>
              <LineChart traces={inputTraces(name)} xDomain={domain} xLabel="時間 (ms)" yLabel={name} />
            </figure>
          {/each}
        </figure>
        {#snippet info()}
          <p>V は圧縮しない（SINDy では状態の 1 列、hybrid では入力として素通しする）ので出さない。</p>
        {/snippet}
      </Card>

      <Card kicker="前処理" title="再構成誤差" subtitle="ゲート → 潜在 → ゲートの RMSE（時刻ごと）">
        <LineChart traces={perComp(reconOf)} xDomain={domain} xLabel="時間 (ms)" yLabel="RMSE" />
        {#snippet info()}
          <p>潜在へ落とした時点で捨てたもの。閉包項をどれだけうまく同定しても、この誤差より下には行かない。</p>
        {/snippet}
      </Card>

      <Card kicker="前処理" title="潜在の軌跡" subtitle={`${pathAxes.x} と ${pathAxes.y}`}>
        {@const traces = pathTraces()}
        <LineChart {traces} xDomain={extent(traces)} xLabel={pathAxes.xIsVoltage ? '膜電位 V (mV)' : pathAxes.x} yLabel={pathAxes.y} />
        {#snippet info()}
          <p>学習ゲートは区画の params に依らないので、区画が違っても同じ多様体に乗るはず。軌跡が重ならなければ、潜在の次元が足りないか、params が混ざっている。</p>
        {/snippet}
      </Card>

      <Card kicker="学習データ" title="電位の分布" subtitle="学習が踏んだ膜電位（区画ごと）">
        {@const traces = coverageTraces()}
        <LineChart {traces} xDomain={extent(traces)} xLabel="膜電位 V (mV)" yLabel="時刻の数" />
        {#snippet info()}
          <p>評価でこの範囲の外へ出た電位は外挿になる。区画を足して学習すると、増えるのは主にこの被覆。</p>
        {/snippet}
      </Card>
    {:else if runId && !error}
      <p class="note">中身を読んでいる。</p>
    {/if}
  {/snippet}

  {#snippet right()}
    {#if model}
      <Card kicker="閉包項" title={sindy ? 'SINDy の係数' : `閉包項（${model.closure.kind}）`} subtitle={sindy ? `0 でない係数 ${nonzero} / ${sindy.xi.length * sindy.features.length}` : ''}>
        {#if sindy}
          <MatrixTable rows={featureHeads()} columns={targetHeads()} values={coefficientRows()} />
        {:else}
          <pre>{JSON.stringify(model.closure, null, 1)}</pre>
        {/if}
        {#snippet info()}
          <p>列が対象変数の時間微分、行がライブラリの項。セルの値が係数 ξ で、色の濃さは log(1 + |ξ|) に比例する。下付きのモデル名は表から省く。</p>
        {/snippet}
      </Card>

      {#if pca}
        <Card kicker="前処理" title="PCA の寄与率" subtitle={`n_components = ${pca.n_components}、累積 ${kept.toFixed(3)}`}>
          <figure>
            <figure>
              <figcaption>成分ごと</figcaption>
              <SignedChart mark="stem" points={ratioPoints()} xDomain={[0.5, pca.explained_variance_ratio.length + 0.5]} yDomain={[0, 1]} xLabel="主成分" yLabel="寄与率" />
            </figure>
            <figure>
              <figcaption>累積</figcaption>
              <LineChart traces={cumulativeTrace()} xDomain={[1, Math.max(2, pca.explained_variance_ratio.length)]} yDomain={[0, 1.02]} xLabel="主成分" yLabel="累積寄与率" guides={[{ label: '保持', tone: 'accent', x: [pca.n_components] }]} />
            </figure>
          </figure>
          {#snippet info()}
            <p>捨てた成分も含めて描く。保持した数が累積の飽和点に対して妥当かを見る。</p>
          {/snippet}
        </Card>
      {:else}
        <Card kicker="前処理" title={`前処理（${model.preprocessor.kind}）`}>
          <pre>{JSON.stringify(model.preprocessor, null, 1)}</pre>
        </Card>
      {/if}
    {/if}
  {/snippet}
</Dashboard>
