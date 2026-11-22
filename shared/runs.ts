// run のファイルの形（計算する機械の Python が書く小さい JSON と .f64）と、その読み方。形式の正本は remote/neurosurrogate/run.py。
// run の中のどこに何があるか（RUN と、.f64 の 1 列を指す compArray など）もここが持つ。読むのは計算する機械の API で、手元は run の形を知らない（API の答えの BenchRow だけが運ばれる）。
// node や bun に依存しない。

/** run の中の .f64 の 1 列。file は run の下のパス、変数ごとに n 個ずつ続いて並んだうちの column 番目。 */
export type ArrayRef = { run: string; file: string; column: number; n: number };

/** run の中の置き場。波形の根 root は、回した波形（学習データの run では学習データ）waves か、原系を列の前処理で射影した projected(区画)、
 * その上の潜在の傾きの正解（射影の前進差分）slope(区画) と、列の学習器の右辺が返す dz/dt の rhs(区画)。
 * decode・plane・compressor・rhs_own・fit は学習器の run の置き場（decode の曲線、ゲートの平面の上の encoder の格子（AE だけ）、圧縮器と右辺に固有の中身、学習データの上の中身）、summary はモデルの run の組んだ演算コスト、kam・equivalent・reduced は KAM のモデルの run の条件と、元のモデルの等価電位の波形と、縮約の波形、spec はどの run も持つ回した条件。 */
export const RUN = {
  waves: 'waves',
  decode: 'compressor/decode.json',
  plane: 'compressor/plane.json',
  compressor: 'compressor/own.json',
  rhs_own: 'rhs/own.json',
  fit: 'fit',
  fitSummary: (compId: number) => `fit/${compId}/summary.json`,
  summary: 'summary.json',
  kam: 'kam.json',
  equivalent: 'equivalent',
  reduced: 'reduced',
  spec: 'spec.json',
  projected: (comp: string) => `projected/${comp}`,
  slope: (comp: string) => `slope/${comp}`,
  rhs: (comp: string) => `rhs/${comp}`,
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

/** 学習器の run の、学習区画 compId の潜在 latent・潜在の傾きの正解 slope（前進差分）・学習器の右辺 rhs（どれも column 列目）か再構成誤差 recon_rmse（1 列）。meta は学習データの目録。 */
export function fitArray(
  run: string,
  meta: WaveMeta,
  compId: number,
  file: 'latent' | 'recon_rmse' | 'slope' | 'rhs',
  column = 0,
): ArrayRef {
  return { run, file: `fit/${compId}/${file}.f64`, column, n: meta.n[0] };
}

/** 学習器の run の decode の曲線（compressor/decode.json）。gates は群のゲート、order は区画の型の状態の並び、axes は潜在の軸ごとの格子 z と、ゲートごとの値（gates の順 × z の点）。 */
export type DecodeJson = {
  gates: string[];
  order: string[];
  axes: { latent: string; z: number[]; gates: number[][] }[];
};
/** 学習器の run のゲートの平面（compressor/plane.json。AE だけ）。群のゲート 2 本 x・y ずつ、格子 xs × ys の上で encoder が返す潜在の値（潜在ごとに ys の行 × xs の列。有限でなければ null）。ほかのゲートは学習データの平均に置いたもの。 */
export type PlaneJson = {
  planes: {
    x: string;
    y: string;
    xs: number[];
    ys: number[];
    latents: { latent: string; values: (number | null)[][] }[];
  }[];
};
/** KAM のモデルの run の条件（kam.json）。comp は描いた区画、absorbed は吸収したゲート、groups は群ごとのゲートと潜在の名前と、Kepler 1992 の式(6)の重みと式(10)の条件。 */
export type KamJson = {
  comp: string;
  absorbed: string[];
  groups: { gates: string[]; latent: string; weights: number[]; criterion: number }[];
};
/** 学習器の run の、圧縮器に固有の中身（compressor/own.json）。 */
export type CompressorJson =
  | { kind: 'pca'; explained_variance_ratio: number[]; n_components: number; components: number[][]; mean: number[] }
  | { kind: 'ae'; hidden: number };
/** 学習器の run の、右辺に固有の中身（rhs/own.json）。 */
export type RhsJson =
  | { kind: 'sindy'; targets: string[]; features: string[]; xi: number[][] }
  | { kind: 'ude'; layers: number[][]; losses: Record<string, number | null> };
/** 学習器の run の、学習区画ごとの全区間のまとめ（fit/<区画の id>/summary.json）。 */
export type FitSummary = { reconstruction_mse: number | null; reconstruction_mse_ratio: number | null };

/** GPU の実測の run の中身。細胞の数ごとのディレクトリ `<細胞の数>/` に、使ったツールの出力を加工せずにツールごとのファイルで置く（remote/neurosurrogate/bench.py と揃える）。
 * own は bench.py が自分で測ったもの（BenchOwn）、jaxPlaced と jaxAfter は入力を置いた後と回した後の JAX の memory_stats、smiApps と smiGpu は nvidia-smi の CSV（このプロセスの使用量と GPU の全容量）、
 * ncu は Nsight Compute の CSV（ncu で測ったときだけ。1 刻みの区画ごとのカーネルの行）。集計は API の bench の口（remote/read/bench.ts）が頼まれるたびに計算する。 */
export const BENCH = {
  own: 'bench.json',
  jaxPlaced: 'jax.memory_stats.placed.json',
  jaxAfter: 'jax.memory_stats.after.json',
  smiApps: 'nvidia-smi.apps.csv',
  smiGpu: 'nvidia-smi.gpu.csv',
  ncu: 'ncu.csv',
  at: (cells: number, file: string) => `${cells}/${file}`,
};

/** bench.py が自分で測ったもの（<細胞の数>/bench.json）。時間はコンパイルの後の 2 回目の実時間。comp_states は区画ごとの、刻みをまたいで持ち越し VRAM に置く状態（V を含む）、
 * decoded は置き換えた区画ごとの、刻みごとに z から作り直すゲート（レジスタに置く）。vram_state_bytes と expected_step_* は状態の置き方から出した予想。
 * spike_counts は系列の点ごとの、同じ電流の細胞の発火の回数（揃っていれば 1 つ）。pid は nvidia-smi の行を引くための、測ったプロセスの pid。 */
export type BenchOwn = {
  cells: number;
  compartments: number;
  steps: number;
  compile_s: number;
  wall_s: number;
  comp_states: Record<string, string[]>;
  decoded: Record<string, string[]>;
  vram_state_bytes: number;
  expected_step_load_bytes: number;
  expected_step_store_bytes: number;
  spike_counts: number[][];
  pid: number;
};

/** GPU の実測の 1 行（API の bench の口の答え）。細胞の数 1 つ。1 刻みがカーネル 1 回で、全細胞をそろえて進める。BenchOwn に、ツールの出力から API が集めた量を足したもの。
 * VRAM の量は JAX（最大・入力・作業量）と nvidia-smi。ncu の量（1 刻みの区画ごとのカーネルを合わせたもの。量は和、レジスタは最大、当たり率は global の読み書きの量で重み付け）は ncu で測ったときだけ。 */
export type BenchRow = Omit<BenchOwn, 'pid'> & {
  cell_steps_per_s: number;
  peak_bytes: number | null;
  input_bytes: number | null;
  work_bytes: number | null;
  smi_used_mib: number | null;
  /** GPU の VRAM の全容量 [バイト]（nvidia-smi）。 */
  vram_total_bytes: number | null;
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
