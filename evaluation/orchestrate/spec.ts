// run の spec.json の形と、段取りと外の画面・図が受け渡す型。何を回すかの宣言は config/ で、読むのは ../experiments/。
// node や bun に依存しない（ブラウザで動く runs.ts も import する）。

/** 1 回のシミュレーションの仕様（Python の SimSpec）。 */
export type SimSpec = {
  target: string;
  current_type: string;
  dt: number;
  current_params: Record<string, number>;
  /** 注入ノード名。省略は soma。 */
  stim?: string;
};

/** 閉包項の項ライブラリの 1 項。type は Python の項カタログの名前。 */
/** gates はレート項をそのゲートのものに絞る（例 'hn'）。省けば群の状態のもの。 */
export type LibrarySpec = { type: string; gates?: string };

/** 1 つの群。gates は区画の状態の名前（Python の cells の states）。params は各 fit 関数の引数へそのまま渡る。 */
export type Group = {
  gates: string[];
  /** kam は学習しない KAM 縮約（rhs も kam、d は 1）。 */
  compressor: 'pca' | 'ae' | 'kam';
  /** 潜在次元。 */
  d: number;
  rhs: 'sindy' | 'ude' | 'kam';
  compressor_params: Record<string, unknown>;
  /** sindy は library_specs と optimizer、ude は学習の hyperparams。 */
  rhs_params: { library_specs?: LibrarySpec[]; [key: string]: unknown };
};

/**
 * 学習の設定（Python の surrogate.Config）。群に入らなかった状態は元の式のまま解く（hybrid）。
 * form が sindy なら膜電位も含めて丸ごと同定し、群は全状態の 1 つだけ。
 */
export type SurrogateConfig = {
  cell: string;
  form: 'hybrid' | 'sindy';
  /** 学習データを取るシミュレーション。 */
  data: SimSpec;
  groups: Group[];
  /** 学習を 1 区画へ絞る名前。null は同じ型の全区画。 */
  train_comp: string | null;
  /** 膜電位へ吸収するゲート（KAM の φ 群、α₀ = 1）。V での定常値に置き、状態を持たない。書かなければ無い。 */
  absorbed?: string[];
};

/** 1 回の掃引実験（Python の EvalSeries）。param が null なら単発（点 1 つ）。 */
export type Series = {
  spec: SimSpec;
  /** 置換するノード名。互換かどうかは Python が名前ごとに検証する。 */
  replace_targets: string[];
  /** 掃引する電流パラメータ名。図の x 軸。 */
  param: string | null;
  values: number[];
};

/** 学習 run 1 本の宣言。name は学習 run の名前（実験の中の `<仕方の名前>_<群分け>` かテストの名前）で、同一性には入らない。 */
export type Train = { name: string; config: SurrogateConfig };

/** GPU の実測 1 つ。model は学習 run の名前（null なら元のモデル）、cells は並べる細胞の数（128 の倍数）、ncu は Nsight Compute で 1 刻みのカーネルを測るか。 */
export type Bench = {
  name: string;
  model: string | null;
  series: string;
  cells: number[];
  ncu: boolean;
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

/** 回す学習 run（名前）と系列の組の並びと、GPU の実測。宣言は使う側が持ち、status に渡す。 */
export type Experiment = { pairs: { model: string; series: string }[]; benches?: Bench[] };

/** spec の形式の版。neurosurrogate/run.py の _FORMAT と揃える。spec に入るので、変えると鍵が変わり古い run は回し直される。 */
export const FORMAT = 7;

/** 回す backend。同じ spec でも計算の順序が変わって結果がずれるので、spec に入れて run を分ける。 */
export type Device = 'cpu' | 'gpu';

/** 入力：電流と掃引。系列から置換範囲を除いたもので、同じ入力の結果は一つのディレクトリにまとまる。 */
export type Input = Omit<Series, 'replace_targets'>;
/** 置き換えたモデル：どの学習 run（ハッシュ）で、どの区画を置き換えるか。 */
export type Model = { train: string; replace_targets: string[] };

/** run.py に渡す spec.json。学習は runs/models/ に、入力 × モデルは runs/inputs/<入力>/ に、GPU の実測は runs/bench/ に置く。model が null なら元のモデル。 */
export type Spec =
  | { format: number; device: Device; kind: 'train'; surrogate: SurrogateConfig }
  | { format: number; device: Device; kind: 'sim'; input: Input; model: Model | null }
  | {
      format: number;
      device: Device;
      kind: 'bench';
      input: Input;
      model: Model | null;
      cells: number[];
      ncu: Bench['ncu'];
    };

/** 回した環境（run.py が meta.json に書く。同一性に入らない）。code と image は段取りが渡した版。 */
export type RunEnv = {
  backend: string;
  device: string;
  jax: string;
  python: string;
  host: string;
  code: string | null;
  image: string | null;
};

/** run の一覧の 1 行（runs/index.json）。id は runs/ の下のパス、kind は run.py が meta.json に書く種類。 */
export type IndexEntry = {
  id: string;
  kind: 'train' | 'original' | 'replaced' | 'bench';
  name: string;
  created: string;
  unit?: string;
  env: RunEnv;
  spec: Spec;
};

/** run 1 本、または組がいまどこまで回っているか。unreplaceable は置換できないと分かった組。 */
export type State = 'done' | 'running' | 'todo' | 'unreplaceable';
/** 外の画面が起こす仕事。spec は回す resolved/ の下のパス。 */
export type Job = { spec: string };
/** 外の画面から起こした仕事 1 つ。failed は終了コードが 0 でないか、終わりを書かずに消えたもの。log はログの末尾の行。 */
export type JobState = { job: Job; started: string; state: 'running' | 'done' | 'failed'; log: string[] };

/** 宣言した run 1 本の名前・場所・状態。id は runs/ の下のパスで、図はここを読む。spec は回す resolved/ の下のパス。 */
export type RunStatus = { name: string; id: string; spec: string; state: State };
/** 実験 1 つの状態。学習 run、系列の元のモデル、組（系列 × 学習 run の置換系）を宣言の順に並べる。state は起こした仕事も見た状態。 */
export type ExperimentStatus = {
  name: string;
  models: { model: string; state: State; run: RunStatus }[];
  series: { series: string; original: RunStatus }[];
  pairs: { series: string; model: string; state: State; run: RunStatus }[];
  benches: { bench: string; state: State; run: RunStatus }[];
};
/** 実験ごとの状態と、起こした仕事。外の画面が status で受け取る。 */
export type Status = { experiments: ExperimentStatus[]; jobs: JobState[] };
