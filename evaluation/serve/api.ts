// 計算する機械の API（serve.ts）の口の形。画面はこの型だけを import して、POST /<口> に JSON を送る。
// 波形は回線に出さない。返すのは run の小さい JSON（meta.json・spikes.json・summary.json・model.json）と、間引いた線・軌跡・分布・指標だけ。
import type { Bin } from '../read/model-json.ts';
import type { CrossingDiff, PairMetrics } from '../read/compare.ts';
import type { IndexEntry, Job, Status } from '../orchestrate/spec.ts';
import type { Judgement } from '../experiments/common.ts';
import type { Point } from '../read/waves.ts';

/** run の中の .f64 の 1 列。file は run の下のパス、変数ごとに n 個ずつ続いて並んだうちの column 番目。 */
export type ArrayRef = { run: string; file: string; column: number; n: number };
/** 時間の図の 1 本。[start, end] ms を buckets 個に区切って最小と最大を残す。t0 と dt は waves/meta.json の値。 */
export type TraceRequest = { array: ArrayRef; t0: number; dt: number; start: number; end: number; buckets: number };
/** 相平面の 1 本。[start, end] ms から等間隔に max 点まで取る。 */
export type PathRequest = { x: ArrayRef; y: ArrayRef; t0: number; dt: number; start: number; end: number; max: number };
/** 系列 1 つの、原系と置き換えた列（run の id）の 1 区画の指標。 */
export type SeriesRequest = { original: string; columns: string[]; comp: string };
/** 1 点の原系との差と Γ（Δ = 2 ms）。 */
export type CrossingRow = CrossingDiff & { gamma: number | null };
/** 系列の指標。キーは列の run の id、並びは掃引の点。times は 0 mV を上向きに越えた時刻（ラスタ）。 */
export type SeriesMetrics = {
  pairs: Record<string, PairMetrics[]>;
  crossings: Record<string, CrossingRow[]>;
  times: { original: number[][]; columns: Record<string, number[][]> };
};

/** 主張 1 つの判定。name は主張の名前、rows は判定の行、判定できなければ null と理由。 */
export type ClaimJudgement = { experiment: string; claim: string; name: string; rows: Judgement[] | null; error?: string };

/** 口ごとの受け取りと返事。 */
export type Api = {
  /** 今の形式の run の一覧（新しい順）。 */
  index: { body: Record<string, never>; reply: IndexEntry[] };
  /** 実験（config/ の宣言）ごとの状態と、起こした仕事。 */
  status: { body: Record<string, never>; reply: Status };
  /** 実験の主張ごとの判定。実験の順、主張の順。 */
  claims: { body: Record<string, never>; reply: ClaimJudgement[] };
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
