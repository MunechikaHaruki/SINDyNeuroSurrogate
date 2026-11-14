// run のファイルの形（計算する機械の Python が書き、API の json の口で手元に返る小さい JSON）と、その読み方。形式の正本は remote/neurosurrogate/run.py。
// run の中のどこに何があるか（RUN と、.f64 の 1 列を指す compArray など）もここが持ち、計算する機械の API と手元の両方が使う。
// node や bun に依存しない（ブラウザで動く画面も import する）。
import type { ArrayRef } from './api.ts';

/** run の中の置き場。波形の根 root は、回した波形 waves、原系を列の前処理で射影した projected(区画)、学習データ train のどれか。 */
export const RUN = {
  waves: 'waves',
  train: 'description/train',
  model: 'description/model.json',
  spikes: 'spikes.json',
  summary: 'summary.json',
  bench: 'bench.json',
  projected: (comp: string) => `projected/${comp}`,
  meta: (root: string) => `${root}/meta.json`,
};

/** 1 列の波形の目録（waves/meta.json）。features は [comp_id, 変数名, ゲートか] の並び。 */
export type WaveMeta = {
  t0: number;
  dt: number;
  n: number[];
  features: [number, string, boolean][];
  nodes: number[] | null;
  names: Record<string, number> | null;
};

/** comp `compId` の変数名（ファイルの中の並び順）。 */
export function variablesOf(meta: WaveMeta, compId: number): string[] {
  const names: string[] = [];
  for (const feature of meta.features) if (feature[0] === compId) names.push(feature[1]);
  return names;
}

/** comp `compId` のゲートの変数名。 */
export function gatesOf(meta: WaveMeta, compId: number): string[] {
  const names: string[] = [];
  for (const feature of meta.features) if (feature[0] === compId && feature[2]) names.push(feature[1]);
  return names;
}

/** 根 root の点 point の区画 compId の変数 1 つ。 */
export function compArray(
  run: string,
  root: string,
  meta: WaveMeta,
  point: number,
  compId: number,
  variable: string,
): ArrayRef {
  const column = variablesOf(meta, compId).indexOf(variable);
  if (column < 0) throw new Error(`${run}/${root} の comp ${compId} に ${variable} が無い`);
  return { run, file: `${root}/${point}/comp${compId}.f64`, column, n: meta.n[point] };
}

/** 根 root の点 point の注入電流 I_ext。 */
export function currentArray(run: string, root: string, meta: WaveMeta, point: number): ArrayRef {
  return { run, file: `${root}/${point}/I_ext.f64`, column: 0, n: meta.n[point] };
}

/** 根 root の点 point の内部電流 I_internal の node 1 つ（結合電流と、刺激した区画では注入電流の和）。 */
export function internalArray(run: string, root: string, meta: WaveMeta, point: number, node: number): ArrayRef {
  const column = (meta.nodes ?? []).indexOf(node);
  return { run, file: `${root}/${point}/I_internal.f64`, column, n: meta.n[point] };
}

/** 学習データ（meta は train の目録）の区画 compId の、同定器への入力 x・潜在 latent・再構成誤差 recon_rmse の column 列目。 */
export function fitArray(
  run: string,
  meta: WaveMeta,
  compId: number,
  file: 'x' | 'latent' | 'recon_rmse',
  column: number,
): ArrayRef {
  return { run, file: `description/fit/${compId}/${file}.f64`, column, n: meta.n[0] };
}

/** 特徴量の名前 → 値の並び（spikes.json の 1 区画・1 点）。null は eFEL の nan。 */
export type Spikes = Record<string, (number | null)[]>;
/** comp 名 → 点ごとの特徴量（spikes.json）。 */
export type SpikesFile = Record<string, Spikes[]>;

/** 閉包項の中身。kind で表現を見分ける。 */
export type ClosureJson =
  | { kind: 'sindy'; targets: string[]; features: string[]; xi: number[][] }
  | { kind: 'ude'; layers: number[][] }
  | { kind: string };
/** 前処理の中身。PCA は捨てた成分も含めた寄与率を持つ。 */
export type PreprocessorJson =
  { kind: 'pca'; explained_variance_ratio: number[]; n_components: number } | { kind: string };
export type NetJson = {
  nodes: { name: string; type: string }[];
  edges: { src: string; dst: string; weight: number }[];
  stim: string;
  replaceable: string[];
  train: string[];
};
/** description/model.json。 */
export type ModelJson = {
  closure: ClosureJson;
  preprocessor: PreprocessorJson;
  net: NetJson;
  train_gates: string[];
  x_names: string[];
  latent_names: string[];
};

/** GPU の実測の結果（bench.json）の 1 行。細胞の数 1 つ。1 刻みがカーネル 1 回で、全細胞をそろえて進める。時間はコンパイルの後の 2 回目の実時間。
 * comp_states は区画ごとの、刻みをまたいで持ち越し、VRAM に置いて毎刻み読み書きする状態（V、潜在 z、縮約しなかったゲート）。
 * decoded は置き換えた区画ごとの、持ち越さず刻みごとに z から作り直すゲート（レジスタに置く）。1 スレッドが 1 区画を持つ。vram_state_bytes は VRAM に置いた状態の総量、
 * expected_step_load_bytes と expected_step_store_bytes は 1 刻みのカーネルが VRAM から読み、書き戻す量の予想（状態・発火の回数・行）。
 * VRAM の量は XLA（最大・入力・作業量）と nvidia-smi。spike_counts は系列の点ごとの、同じ電流の細胞の発火の回数（揃っていれば 1 つ）。
 * ncu の量（1 刻みのカーネル 1 回分）は ncu を回したときだけ。 */
export type BenchRow = {
  cells: number;
  steps: number;
  compile_s: number;
  wall_s: number;
  cell_steps_per_s: number;
  /** 区画の数と、区画の名前 → VRAM に置く変数（V を含む）、置き換えた区画の名前 → z から作り直すゲート。 */
  compartments: number;
  comp_states: Record<string, string[]>;
  decoded: Record<string, string[]>;
  vram_state_bytes: number;
  expected_step_load_bytes: number;
  expected_step_store_bytes: number;
  peak_bytes: number | null;
  input_bytes: number | null;
  work_bytes: number | null;
  smi_used_mib: number | null;
  /** GPU の VRAM の全容量 [バイト]（nvidia-smi）。 */
  vram_total_bytes: number | null;
  spike_counts: number[][];
  registers_per_thread?: number | null;
  threads?: number | null;
  global_store_bytes?: number | null;
  global_load_bytes?: number | null;
  local_store_bytes?: number | null;
  local_load_bytes?: number | null;
  dram_write_bytes?: number | null;
  dram_read_bytes?: number | null;
  l1_hit_pct?: number | null;
  l2_hit_pct?: number | null;
};
