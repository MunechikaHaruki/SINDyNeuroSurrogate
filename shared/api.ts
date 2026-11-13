// 帰りの JSON の形。計算する機械の API（remote/serve/serve.ts）の口ごとの受け取りと返事で、画面はこの型だけを import して POST /<口> に JSON を送る。
// 波形は回線に出さない。返すのは run の小さい JSON（meta.json・spikes.json・summary.json・model.json）と、間引いた線・軌跡・分布・指標だけ。
import type { IndexEntry, Job, Status } from './spec.ts';

/** 線の 1 点。 */
export type Point = { x: number; y: number };
/** 分布の 1 区間。 */
export type Bin = { center: number; count: number };
/** 1 ペア（原系, 置換系）の 1 comp の指標。null は値が無い（nan）。 */
export type PairMetrics = {
  rows: Record<string, [number | null, number | null]>;
  scalars: Record<string, number | null>;
  spikes: Record<string, [(number | null)[], (number | null)[]]>;
  diverged: boolean;
};
/** 元のモデルとの差。peak は山の平均の差（モデル − 元）、lastShift は両方にある最後の山の時刻の差の大きさ。 */
export type CrossingDiff = { count: [number, number]; peak: number | null; lastShift: number | null };

/** run の中の .f64 の 1 列。file は run の下のパス、変数ごとに n 個ずつ続いて並んだうちの column 番目。 */
export type ArrayRef = { run: string; file: string; column: number; n: number };
/** 時間の図の 1 本。[start, end] ms を buckets 個に区切って最小と最大を残す。t0 と dt は waves/meta.json の値。 */
export type TraceRequest = { array: ArrayRef; t0: number; dt: number; start: number; end: number; buckets: number };
/** 相平面の 1 本。[start, end] ms から等間隔に max 点まで取る。 */
export type PathRequest = { x: ArrayRef; y: ArrayRef; t0: number; dt: number; start: number; end: number; max: number };
/** 系列 1 つの、原系と置き換えた列（run の id）の 1 区画の指標。delta は Γ の許容幅 Δ [ms]（判定の物差しは頼む側が持つ）。 */
export type SeriesRequest = { original: string; columns: string[]; comp: string; delta: number };
/** 1 点の原系との差と Γ（Δ は頼んだ delta）。 */
export type CrossingRow = CrossingDiff & { gamma: number | null };
/** 系列の指標。キーは列の run の id、並びは掃引の点。times は 0 mV を上向きに越えた時刻（ラスタ）。 */
export type SeriesMetrics = {
  pairs: Record<string, PairMetrics[]>;
  crossings: Record<string, CrossingRow[]>;
  times: { original: number[][]; columns: Record<string, number[][]> };
};

/** 口ごとの受け取りと返事。 */
export type Api = {
  /** 今の形式の run の一覧（新しい順）。 */
  index: { body: Record<string, never>; reply: IndexEntry[] };
  /** 実験（resolved/ の下のディレクトリ）ごとの状態と、起こした仕事。 */
  status: { body: Record<string, never>; reply: Status };
  /** 学習か評価を切り離して回す。同じ組が回っていれば起こさない。 */
  launch: { body: { job: Job }; reply: { started: boolean } };
  /** run の中の .json 1 つ。 */
  json: { body: { run: string; path: string }; reply: unknown };
  traces: { body: { requests: TraceRequest[] }; reply: Point[][] };
  paths: { body: { requests: PathRequest[] }; reply: Point[][] };
  /** 列ごとの分布。区間は全部の列の有限な値の最小から最大で揃える。 */
  histograms: { body: { arrays: ArrayRef[]; bins: number }; reply: Bin[][] };
  series: { body: SeriesRequest; reply: SeriesMetrics };
};
