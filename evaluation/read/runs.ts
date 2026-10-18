// SINDyNeuroSurrogate の結果を runs/ から読む（評価の冊と学習 run の冊が共有する）。
// run は段取り（orchestrate.ts。操作盤の冊のボタンから起こす）が runs/<ハッシュ>/ に置き、一覧を runs/index.json に書く。
// spec の形は conf.ts、ファイルの形式は neurosurrogate/run.py の先頭。ここは読むだけで、Python は import しない。配信は context-router の開発サーバー。
import { FORMAT, type IndexEntry, type Job, type Status } from '../orchestrate/conf.ts';

/** runs/ の置き場。この .ts から見た相対パスを開発サーバーの URL に直す（文字列を直に渡すと Vite が資産として扱う）。 */
const RUNS_DIR = '../../runs/';
export const RUNS = new URL(RUNS_DIR, import.meta.url).href;

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
/** 評価した系列。原系の run と、どこを置換したかの組。 */
export type SeriesEntry = {
  key: string;
  name: string;
  /** 原系の run のハッシュ */
  run: string;
  param: string | null;
  values: number[];
  unit: string;
  replaceTargets: string[];
};
/** 系列を 1 つの学習 run で置換して回した列。 */
export type Column = { run: string; train: string; trainName: string };
export type TrainRun = { hash: string; name: string };
/** 1 ペア（原系, 置換系）の 1 comp の指標。null は値が無い（nan）。 */
export type PairMetrics = {
  rows: Record<string, [number | null, number | null]>;
  scalars: Record<string, number | null>;
  spikes: Record<string, [number[], number[]]>;
  diverged: boolean;
};
/** comp 名 → 点ごとの指標（metrics.json）。 */
export type Metrics = Record<string, PairMetrics[]>;

/** run の一覧（新しい順）。今の形式の run だけ。まだ何も回していなければ空。 */
export async function listRuns(): Promise<IndexEntry[]> {
  const response = await fetch(`${RUNS}index.json`, { cache: 'no-store' });
  if (response.status === 404) return [];
  if (!response.ok) throw new Error(`index.json: ${response.status}`);
  const list: IndexEntry[] = [];
  for (const run of (await response.json()) as IndexEntry[]) if (run.spec.format === FORMAT) list.push(run);
  return list;
}

/** 系列の鍵。原系が同じでも置換範囲が違えば別の系列。 */
function seriesKey(original: string, targets: string[]): string {
  return `${original}/${targets.join(',')}`;
}

/** 評価した系列。置換系の run を原系と置換範囲で束ね、置換系がまだ無い原系も足す。 */
export function listSeries(runs: IndexEntry[]): SeriesEntry[] {
  const list: SeriesEntry[] = [];
  const keys = new Set<string>();
  const used = new Set<string>();
  for (const run of runs) {
    const spec = run.spec;
    if (spec.kind !== 'replaced') continue;
    const key = seriesKey(spec.original, spec.series.replace_targets);
    if (keys.has(key)) continue;
    keys.add(key);
    used.add(spec.original);
    list.push({ key, name: run.name, run: spec.original, param: spec.series.param, values: spec.series.values, unit: run.unit ?? '', replaceTargets: spec.series.replace_targets });
  }
  for (const run of runs) {
    const spec = run.spec;
    if (spec.kind !== 'original' || used.has(run.hash)) continue;
    list.push({ key: seriesKey(run.hash, []), name: run.name, run: run.hash, param: spec.series.param, values: spec.series.values, unit: run.unit ?? '', replaceTargets: [] });
  }
  return list;
}

/** 系列を置換して回した列。 */
export function listColumns(runs: IndexEntry[], entry: SeriesEntry): Column[] {
  const names: Record<string, string> = {};
  for (const run of runs) if (run.kind === 'train') names[run.hash] = run.name;
  const list: Column[] = [];
  for (const run of runs) {
    const spec = run.spec;
    if (spec.kind !== 'replaced' || seriesKey(spec.original, spec.series.replace_targets) !== entry.key) continue;
    list.push({ run: run.hash, train: spec.train, trainName: names[spec.train] ?? spec.train });
  }
  return list;
}

/** 学習を終えた run。 */
export function listTrainRuns(runs: IndexEntry[]): TrainRun[] {
  const list: TrainRun[] = [];
  for (const run of runs) if (run.kind === 'train') list.push({ hash: run.hash, name: run.name });
  return list;
}

/** 操作盤が学習と評価を起こす口（api.ts）。 */
const API = '/api/face/research/50-59-research/53-evaluation/orchestrate';

export async function readStatus(): Promise<Status> {
  const response = await fetch(API);
  if (!response.ok) throw new Error(`${API}: ${response.status}`);
  return (await response.json()) as Status;
}

/** 学習か評価を切り離して回す。同じ組が回っていれば何もしない。 */
export async function start(job: Job): Promise<void> {
  const response = await fetch(API, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(job) });
  if (!response.ok) throw new Error(`${API}: ${((await response.json()) as { message: string }).message}`);
}

/** 回っているものの数。減ったら何かが回し終わったので、run の一覧を読み直す。 */
export function runningCount(status: Status): number {
  let count = 0;
  for (const train of status.trains) if (train.state === 'running') count++;
  for (const pair of status.pairs) if (pair.state === 'running') count++;
  return count;
}

export function fileUrl(run: string, path: string): string {
  return `${RUNS}${run}/${path}`;
}

export async function readJson<T>(run: string, path: string): Promise<T> {
  const response = await fetch(fileUrl(run, path));
  if (!response.ok) throw new Error(`${run}/${path}: ${response.status}`);
  return (await response.json()) as T;
}

async function readBuffer(run: string, path: string): Promise<ArrayBuffer> {
  const response = await fetch(fileUrl(run, path));
  if (!response.ok) throw new Error(`${run}/${path}: ${response.status}`);
  return await response.arrayBuffer();
}

/** little-endian の float32 を並べただけのファイル（`.f32`）。 */
export async function readF32(run: string, path: string): Promise<Float32Array> {
  return new Float32Array(await readBuffer(run, path));
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

/** 読んだ波形。run は書き換わらないので、選び直すたびに読み直さない。 */
const loaded = new Map<string, Promise<Record<string, Float32Array>>>();

/** 1 点の 1 comp の全変数。root は waves か projected/<comp 名>。 */
export function readComp(
  run: string,
  root: string,
  meta: WaveMeta,
  point: number,
  compId: number,
): Promise<Record<string, Float32Array>> {
  const path = `${root}/${point}/comp${compId}.f32`;
  const key = `${run}/${path}`;
  if (!loaded.has(key)) {
    const reading = readBuffer(run, path).then(buffer =>
      decodeComp(buffer, variablesOf(meta, compId), meta.n[point]),
    );
    loaded.set(key, reading);
  }
  return loaded.get(key)!;
}

/** 1 点の内部電流 I_internal（結合電流と、刺激した区画では注入電流の和）。キーは node の id（comp の id と同じ）。 */
export function readInternal(
  run: string,
  root: string,
  meta: WaveMeta,
  point: number,
): Promise<Record<string, Float32Array>> {
  const path = `${root}/${point}/I_internal.f32`;
  const key = `${run}/${path}`;
  if (!loaded.has(key)) {
    const nodes: string[] = [];
    for (const node of meta.nodes ?? []) nodes.push(String(node));
    loaded.set(
      key,
      readBuffer(run, path).then(buffer => decodeComp(buffer, nodes, meta.n[point])),
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
export async function readCurrent(run: string, root: string, point: number): Promise<Float32Array> {
  return readF32(run, `${root}/${point}/I_ext.f32`);
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
