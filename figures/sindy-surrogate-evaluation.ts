// SINDyNeuroSurrogate の結果を、MLflow の REST から読む（評価の冊と学習 run の冊が共有する）。
// 評価はこの repo の `just eval` が回し、波形 experiment に 1 列 1 run で置く（形式は scripts/mlflow_io/_files.py）。
// ここは読むだけで、研究コードは import しない。サーバーはこの repo の `just mlflow`（Mac ではログイン時に常駐）。

export const MLFLOW = 'http://127.0.0.1:5100';
const EVAL_EXPERIMENT = 'eval_series';
const TRAIN_EXPERIMENT = 'test_static_params';

export type Point = { x: number; y: number };
/** 1 列の波形の目録（waves/meta.json）。features は [comp_id, 変数名, ゲートか] の並び。 */
export type WaveMeta = {
  t0: number;
  dt: number;
  n: number[];
  features: [number, string, boolean][];
  nodes: number[] | null;
  names: Record<string, number> | null;
};
/** 評価した系列。原系の run が持つ。 */
export type SeriesEntry = {
  name: string;
  runId: string;
  hash: string;
  param: string | null;
  values: number[];
  unit: string;
  replaceTargets: string[];
};
/** 系列を 1 つの学習 run で置換して回した列。 */
export type Column = { evalRunId: string; trainRunId: string; trainName: string };
export type TrainRun = { runId: string; name: string };
/** 1 ペア（原系, 置換系）の 1 comp の指標。null は値が無い（nan）。 */
export type PairMetrics = {
  rows: Record<string, [number | null, number | null]>;
  scalars: Record<string, number | null>;
  spikes: Record<string, [number[], number[]]>;
  diverged: boolean;
};
/** comp 名 → 点ごとの指標（metrics.json）。 */
export type Metrics = Record<string, PairMetrics[]>;

type Entry = { key: string; value: string };
type Run = { info: { run_id: string; run_name?: string }; data: { params?: Entry[]; tags?: Entry[] } };

/** run の param か tag の値。無ければ undefined。 */
function valueOf(entries: Entry[] | undefined, key: string): string | undefined {
  for (const entry of entries ?? []) if (entry.key === key) return entry.value;
  return undefined;
}

async function request<T>(path: string, body?: unknown): Promise<T> {
  const init: RequestInit =
    body === undefined
      ? {}
      : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
  const response = await fetch(`${MLFLOW}${path}`, init);
  if (!response.ok) throw new Error(`MLflow ${path}: ${response.status}`);
  return (await response.json()) as T;
}

/** experiment の id。無ければ undefined（まだ一度も評価していない）。 */
async function experimentId(name: string): Promise<string | undefined> {
  const response = await fetch(
    `${MLFLOW}/api/2.0/mlflow/experiments/get-by-name?experiment_name=${encodeURIComponent(name)}`,
  );
  if (response.status === 404) return undefined;
  if (!response.ok) throw new Error(`MLflow experiment ${name}: ${response.status}`);
  const body = (await response.json()) as { experiment: { experiment_id: string } };
  return body.experiment.experiment_id;
}

/** 条件に合う run を新しい順に。 */
async function searchRuns(experiment: string, filter: string): Promise<Run[]> {
  const id = await experimentId(experiment);
  if (id === undefined) return [];
  const body = await request<{ runs?: Run[] }>('/api/2.0/mlflow/runs/search', {
    experiment_ids: [id],
    filter,
    max_results: 1000,
    order_by: ['attributes.start_time DESC'],
  });
  return body.runs ?? [];
}

/** 評価した系列の一覧。同じ系列を回し直した run があれば新しい方だけ。 */
export async function listSeries(): Promise<SeriesEntry[]> {
  const list: SeriesEntry[] = [];
  const seen = new Set<string>();
  for (const run of await searchRuns(EVAL_EXPERIMENT, "tags.kind = 'original'")) {
    const hash = valueOf(run.data.tags, 'series_hash')!;
    if (seen.has(hash)) continue;
    seen.add(hash);
    const series = JSON.parse(valueOf(run.data.params, 'series')!) as {
      param: string | null;
      values: number[];
      replace_targets: string[];
    };
    list.push({
      name: valueOf(run.data.tags, 'series_name') ?? run.info.run_name ?? hash,
      runId: run.info.run_id,
      hash,
      param: series.param,
      values: series.values,
      unit: valueOf(run.data.params, 'unit') ?? '',
      replaceTargets: series.replace_targets,
    });
  }
  return list;
}

/** 学習 run の名前。 */
async function runName(runId: string): Promise<string> {
  const body = await request<{ run: Run }>(`/api/2.0/mlflow/runs/get?run_id=${runId}`);
  return body.run.info.run_name ?? runId;
}

/** 系列（原系の鍵 hash）を置換して回した列。学習 run ごとに新しい方だけ。 */
export async function listColumns(hash: string): Promise<Column[]> {
  const list: Column[] = [];
  const seen = new Set<string>();
  for (const run of await searchRuns(EVAL_EXPERIMENT, `tags.kind = 'surrogate' and tags.original_hash = '${hash}'`)) {
    const trainRunId = valueOf(run.data.params, 'run_id')!;
    if (seen.has(trainRunId)) continue;
    seen.add(trainRunId);
    list.push({ evalRunId: run.info.run_id, trainRunId, trainName: await runName(trainRunId) });
  }
  return list;
}

/** 学習を終えた run。まだ評価していない run に `just eval` を案内するために使う。 */
export async function listTrainRuns(): Promise<TrainRun[]> {
  const list: TrainRun[] = [];
  for (const run of await searchRuns(TRAIN_EXPERIMENT, "attributes.status = 'FINISHED'")) {
    list.push({ runId: run.info.run_id, name: run.info.run_name ?? run.info.run_id });
  }
  return list;
}

export function artifactUrl(runId: string, path: string): string {
  return `${MLFLOW}/get-artifact?path=${encodeURIComponent(path)}&run_uuid=${runId}`;
}

export async function readJson<T>(runId: string, path: string): Promise<T> {
  const response = await fetch(artifactUrl(runId, path));
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return (await response.json()) as T;
}

async function readBuffer(runId: string, path: string): Promise<ArrayBuffer> {
  const response = await fetch(artifactUrl(runId, path));
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return await response.arrayBuffer();
}

/** little-endian の float32 を並べただけのファイル（`.f32`）。 */
export async function readF32(runId: string, path: string): Promise<Float32Array> {
  return new Float32Array(await readBuffer(runId, path));
}

/** run の `path` の下にあるもの。無ければ空。 */
export async function listArtifacts(runId: string, path: string): Promise<string[]> {
  const response = await fetch(
    `${MLFLOW}/api/2.0/mlflow/artifacts/list?run_id=${runId}&path=${encodeURIComponent(path)}`,
  );
  if (!response.ok) return [];
  const body = (await response.json()) as { files?: { path: string }[] };
  const paths: string[] = [];
  for (const file of body.files ?? []) paths.push(file.path);
  return paths;
}

/** comp `compId` の変数名（ファイルの中の並び順）。 */
export function variablesOf(meta: WaveMeta, compId: number): string[] {
  const names: string[] = [];
  for (const feature of meta.features) if (feature[0] === compId) names.push(feature[1]);
  return names;
}

/** comp<id>.f32 を変数ごとの列に分ける。変数ごとに n 個ずつ続いて並んでいる。 */
export function decodeComp(buffer: ArrayBuffer, variables: string[], n: number): Record<string, Float32Array> {
  const all = new Float32Array(buffer);
  if (all.length !== variables.length * n)
    throw new Error(`comp の長さ ${all.length} が ${variables.length} 変数 × ${n} 点と合わない`);
  const table: Record<string, Float32Array> = {};
  for (let i = 0; i < variables.length; i++) table[variables[i]] = all.subarray(i * n, (i + 1) * n);
  return table;
}

/** 読んだ波形。run の artifact は書き換わらないので、選び直すたびに読み直さない。 */
const loaded = new Map<string, Promise<Record<string, Float32Array>>>();

/** 1 点の 1 comp の全変数。root は waves か projected/<comp 名>。 */
export function readComp(
  runId: string,
  root: string,
  meta: WaveMeta,
  point: number,
  compId: number,
): Promise<Record<string, Float32Array>> {
  const path = `${root}/${point}/comp${compId}.f32`;
  const key = `${runId}/${path}`;
  if (!loaded.has(key)) {
    const reading = readBuffer(runId, path).then(buffer =>
      decodeComp(buffer, variablesOf(meta, compId), meta.n[point]),
    );
    loaded.set(key, reading);
  }
  return loaded.get(key)!;
}

/** 1 点の内部電流 I_internal（結合電流と、刺激した区画では注入電流の和）。キーは node の id（comp の id と同じ）。 */
export function readInternal(
  runId: string,
  root: string,
  meta: WaveMeta,
  point: number,
): Promise<Record<string, Float32Array>> {
  const path = `${root}/${point}/I_internal.f32`;
  const key = `${runId}/${path}`;
  if (!loaded.has(key)) {
    const nodes: string[] = [];
    for (const node of meta.nodes ?? []) nodes.push(String(node));
    loaded.set(
      key,
      readBuffer(runId, path).then(buffer => decodeComp(buffer, nodes, meta.n[point])),
    );
  }
  return loaded.get(key)!;
}

/** comp `compId` のゲートの変数名。 */
export function gatesOf(meta: WaveMeta, compId: number): string[] {
  const names: string[] = [];
  for (const feature of meta.features) if (feature[0] === compId && feature[2]) names.push(feature[1]);
  return names;
}

/** 1 点の注入電流 I_ext。 */
export async function readCurrent(runId: string, root: string, point: number): Promise<Float32Array> {
  return readF32(runId, `${root}/${point}/I_ext.f32`);
}

/**
 * 表示用に間引く。[start, end] ms を buckets 個に区切り、区間ごとに最小と最大を時刻順に残す。
 * スパイクの山と谷は潰れずに残り、点の数は 2 × buckets 以下になる。有限でない値は捨てる。
 */
export function decimate(
  values: Float32Array,
  t0: number,
  dt: number,
  start: number,
  end: number,
  buckets: number,
): Point[] {
  const points: Point[] = [];
  const first = Math.max(0, Math.ceil((start - t0) / dt));
  const last = Math.min(values.length - 1, Math.floor((end - t0) / dt));
  const width = (last - first + 1) / buckets;
  for (let b = 0; b < buckets; b++) {
    const from = first + Math.floor(b * width);
    const to = Math.min(last, first + Math.floor((b + 1) * width) - 1);
    let low = -1;
    let high = -1;
    for (let i = from; i <= to; i++) {
      if (!Number.isFinite(values[i])) continue;
      if (low < 0 || values[i] < values[low]) low = i;
      if (high < 0 || values[i] > values[high]) high = i;
    }
    if (low < 0) continue;
    const earlier = Math.min(low, high);
    const later = Math.max(low, high);
    points.push({ x: t0 + earlier * dt, y: values[earlier] });
    if (later !== earlier) points.push({ x: t0 + later * dt, y: values[later] });
  }
  return points;
}
