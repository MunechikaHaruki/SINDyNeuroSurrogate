// run の spec.json の形と、段取り・操作盤の口・図が受け渡す型。何を回すかの宣言は conf.ts。
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
  compressor: 'pca' | 'ae';
  /** 潜在次元。 */
  d: number;
  rhs: 'sindy' | 'ude';
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

/** 学習 run 1 本の宣言。name は図に出す名前で、同一性には入らない。 */
export type Train = { name: string; config: SurrogateConfig };

/** 見比べる preset と系列の組。 */
export type Experiment = { presets: string[]; series: string[] };

/** spec の形式の版。neurosurrogate/run.py の _FORMAT と揃える。spec に入るので、変えると鍵が変わり古い run は回し直される。 */
export const FORMAT = 6;

/** 入力：電流と掃引。系列から置換範囲を除いたもので、同じ入力の結果は一つのディレクトリにまとまる。 */
export type Input = Omit<Series, 'replace_targets'>;
/** 置き換えたモデル：どの学習 run（ハッシュ）で、どの区画を置き換えるか。 */
export type Model = { train: string; replace_targets: string[] };

/** run.py に渡す spec.json。学習は runs/models/ に、入力 × モデルは runs/inputs/<入力>/ に置く。model が null なら元のモデル。 */
export type Spec =
  | { format: number; kind: 'train'; surrogate: SurrogateConfig }
  | { format: number; kind: 'sim'; input: Input; model: Model | null };

/** run の一覧の 1 行（runs/index.json）。id は runs/ の下のパス、kind は run.py が meta.json に書く種類。 */
export type IndexEntry = { id: string; kind: 'train' | 'original' | 'replaced'; name: string; created: string; unit?: string; spec: Spec };

/** run 1 本、または組がいまどこまで回っているか。unreplaceable は置換できないと分かった組。 */
export type State = 'done' | 'running' | 'todo' | 'unreplaceable';
/** 操作盤のボタンが起こす仕事。 */
export type Job = { action: 'train'; preset: string } | { action: 'eval'; series: string; preset: string };
/** 操作盤から起こした仕事 1 つ。failed は終了コードが 0 でないか、終わりを書かずに消えたもの。log はログの末尾の行。 */
export type JobState = { job: Job; started: string; state: 'running' | 'done' | 'failed'; log: string[] };

/** 宣言した run 1 本（学習なら掃引の 1 点）の名前・場所・状態。id は runs/ の下のパスで、図はここを読む。 */
export type RunStatus = { name: string; id: string; state: State };
/** 実験 1 つの状態。preset の runs は掃引の点の順、組の runs はその preset の runs と同じ順。 */
export type ExperimentStatus = {
  name: string;
  presets: { preset: string; state: State; runs: RunStatus[] }[];
  series: { series: string; original: RunStatus }[];
  pairs: { series: string; preset: string; state: State; runs: RunStatus[] }[];
};
/** 実験ごとの状態と、起こした仕事。操作盤の口（api.ts）が返す。 */
export type Status = { experiments: ExperimentStatus[]; jobs: JobState[] };
