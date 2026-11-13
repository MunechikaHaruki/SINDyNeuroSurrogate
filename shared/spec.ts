// 行きの JSON（resolved/ の spec.json）の形と、計算する機械が返す run の一覧と状態の形。手元（../local/）と計算する機械（../remote/）が受け渡すのはこの形と隣の api.ts・runs.ts だけ。
// node や bun に依存しない（ブラウザで動く画面も import する）。

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

/** spec の形式の版。neurosurrogate/run.py の _FORMAT と揃える。spec に入るので、変えると鍵が変わり古い run は回し直される。 */
export const FORMAT = 7;

/** 回す backend。同じ spec でも計算の順序が変わって結果がずれるので、spec に入れて run を分ける。 */
export type Device = 'cpu' | 'gpu';

/** 入力：電流と掃引（param が null なら単発で点 1 つ）。同じ入力の結果は一つのディレクトリにまとまる。 */
export type Input = {
  spec: SimSpec;
  /** 掃引する電流パラメータ名。図の x 軸。 */
  param: string | null;
  values: number[];
};
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
      /** Nsight Compute で 1 刻みのカーネルを測るか。 */
      ncu: boolean;
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

/** resolved/ の spec 1 つの名前・場所・状態。id は runs/ の下のパスで、図はここを読む。spec は回す resolved/ の下のパス。 */
export type RunStatus = { name: string; id: string; spec: string; state: State };
/** 実験（resolved/ の下のディレクトリ）1 つの状態。学習 run、系列の元のモデル、組（系列 × 学習 run の置換系）を名前の順に並べる（宣言の順は手元が持つ）。state は起こした仕事も見た状態。 */
export type ExperimentStatus = {
  name: string;
  models: { model: string; state: State; run: RunStatus }[];
  series: { series: string; original: RunStatus }[];
  pairs: { series: string; model: string; state: State; run: RunStatus }[];
  benches: { bench: string; state: State; run: RunStatus }[];
};
/** 実験ごとの状態と、起こした仕事。外の画面が status で受け取る。 */
export type Status = { experiments: ExperimentStatus[]; jobs: JobState[] };
