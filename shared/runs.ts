// run のファイルの形（計算する機械の Python が書く小さい JSON と .f64）と、その読み方。形式の正本は remote/neurosurrogate/run.py。
// run の中のどこに何があるか（RUN と、.f64 の 1 列を指す compArray など）もここが持つ。読むのは計算する機械の API で、手元は run の形を知らない（API の答えの BenchRow だけが運ばれる）。
// node や bun に依存しない。

/** run の中の .f64 の 1 列。file は run の下のパス、変数ごとに n 個ずつ続いて並んだうちの column 番目。 */
export type ArrayRef = { run: string; file: string; column: number; n: number };

/** run の中の置き場。波形の根 root は、回した波形（学習データの run では学習データ）waves か、原系を列の前処理で射影した projected(区画)。
 * part は学習器の run の中身、fit はその学習区画の目録、summary はモデルの run の組んだ演算コスト、spec はどの run も持つ回した条件。 */
export const RUN = {
  waves: 'waves',
  part: 'part.json',
  fit: 'fit/meta.json',
  summary: 'summary.json',
  spec: 'spec.json',
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

/** 学習器の run の fit/meta.json。学習データの t0・dt・長さ n と、学習区画の名前 → id。 */
export type FitMeta = { t0: number; dt: number; n: number; names: Record<string, number> };

/** 学習器の run の、学習区画 compId の学習データの群の状態 gates（column 列目）・膜電位 V（1 列）、潜在 latent（column 列目）か再構成誤差 recon_rmse（1 列）。 */
export function fitArray(
  run: string,
  meta: FitMeta,
  compId: number,
  file: 'gates' | 'V' | 'latent' | 'recon_rmse',
  column = 0,
): ArrayRef {
  return { run, file: `fit/${compId}/${file}.f64`, column, n: meta.n };
}

/** 閉包項の中身。kind で表現を見分ける。 */
export type ClosureJson =
  | { kind: 'sindy'; targets: string[]; features: string[]; xi: number[][] }
  | { kind: 'ude'; layers: number[][] }
  | { kind: string };
/** 前処理の中身。PCA は捨てた成分も含めた寄与率を持つ。 */
export type PreprocessorJson =
  { kind: 'pca'; explained_variance_ratio: number[]; n_components: number } | { kind: string };
/** 学習器の run の中身（part.json）。latents は潜在の名前（群の名前で修飾し学習器の中で z1 から。h+n.z1。どのモデルの波形でも同じ名前）、comps は学習区画の名前（fit/<区画の id>/）。 */
export type PartJson = {
  gates: string[];
  latents: string[];
  comps: string[];
  closure: ClosureJson;
  preprocessor: PreprocessorJson;
  metrics: Record<string, number | null>;
};

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
