// 行きの JSON（resolved/ の spec.json と、宣言の写し plans.json）の形と、計算する機械が返す run の一覧と状態の形。手元（../local/）と計算する機械（../remote/）が受け渡すのはこの形と隣の api.ts・runs.ts だけ。
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

/**
 * spec の形式の版。spec の種類ごとに持ち、neurosurrogate/run.py の _FORMATS と揃える（remote/run/places.test.ts が確かめる）。
 * spec に入るので、変えるとその種類の鍵が変わり、その run と、ハッシュでそれを指す後ろの run（学習器なら、モデル・置換系・GPU の実測）だけが回し直される。
 */
export const FORMATS = { part: 23, model: 14, sim: 13, bench: 11 } as const;

/** 回す backend。同じ spec でも計算の順序が変わって結果がずれるので、spec に入れて run を分ける。 */
export type Device = 'cpu' | 'gpu';

/** 入力：電流と掃引（param が null なら単発で点 1 つ）。同じ入力の結果は一つのディレクトリにまとまる。 */
export type Input = {
  spec: SimSpec;
  /** 掃引する電流パラメータ名。図の x 軸。 */
  param: string | null;
  values: number[];
};
/** 置き換えたモデル：どのモデル（runs/models/ のハッシュ）で、どの区画を置き換えるか。 */
export type Model = { surrogate: string; replace_targets: string[] };

/**
 * run.py に渡す spec.json。学習の入力の元のモデル → 学習器 → モデル → シミュレーションの順に、後ろの spec が前の run を鍵で指す。
 * 学習器（群 1 つの圧縮器と右辺）は runs/parts/ に、学習器の並びを組んだモデル（KAM は学習器を持たず、群と学習の入力を指す）は runs/models/ に、
 * 入力 × モデルは runs/inputs/<入力の鍵>/ に、GPU の実測は runs/bench/ に置く。sim と bench の model が null なら元のモデル。
 * 学習データは置き場を持たず、学習の入力（学習用の電流の単発の入力）の元のモデル runs/inputs/<入力の鍵>/original/ がそれにあたる。
 */
export type Spec =
  | {
      format: number;
      device: Device;
      kind: 'part';
      cell: string;
      form: SurrogateConfig['form'];
      group: Group;
      /** 学習の入力の鍵（runs/inputs/<鍵>/original/ が学習データ）と、学習を絞る区画。 */
      input: string;
      train_comp: string | null;
    }
  | {
      format: number;
      device: Device;
      kind: 'model';
      cell: string;
      form: SurrogateConfig['form'];
      /** 膜電位へ吸収するゲート。書かなければ無い。 */
      absorbed?: string[];
      /** 学習器（runs/parts/）のハッシュの並び（群の順）。KAM のモデルは空。 */
      parts: string[];
      /** KAM のモデル（Kepler 1992 の縮約。学習せず、区画の型の式から群を等価電位 1 本ずつにまとめる）だけが持つ。groups は群（圧縮器と右辺はどちらも kam）、
       * input と train_comp は、元のモデルと縮約したモデルの等価電位と膜電位を描く学習の入力の鍵（runs/inputs/<鍵>/original/）と区画。 */
      kam?: { groups: Group[]; input: string; train_comp: string | null };
    }
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

/** 宣言の系列 1 つ。入力と、置き換える区画。 */
export type PlanSeries = Input & { replace_targets: string[] };
/** 宣言のモデル 1 つ。name は `<仕方>.<群分け>`、method と grouping はその仕方と群分けの名前。 */
export type PlanModel = { name: string; method: string; grouping: string; config: SurrogateConfig };
/** 宣言の GPU の実測 1 つ。model が null なら元のモデル。 */
export type PlanBench = { name: string; model: string | null; series: string; cells: number[]; ncu: boolean };
/** 宣言の実験 1 つ。groupings と methods は画面が絞り込みに使う名前。モデルと系列の名前は実験の中で一意。 */
export type PlanEntry = {
  id: string;
  group: string;
  description: string;
  groupings: string[];
  methods: string[];
  models: PlanModel[];
  series: Record<string, PlanSeries>;
  pairs: { model: string; series: string }[];
  benches: PlanBench[];
};
/**
 * 宣言の写し（resolved/plans.json）。手元が宣言から書き、計算する機械が実験の画面の答え（設定、目次、並びの順）を組むのに読む。run の同一性には入らない。
 * groups と experiments は宣言の順、measures は比べ方の物差し（Γ の許容幅 Δ [ms]）。
 */
export type Plans = {
  groups: { name: string; description: string }[];
  experiments: PlanEntry[];
  measures: { coincidence_window: number };
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

/** run の一覧の 1 行（API の index）。id は runs/ の下のパス、kind は run.py が meta.json に書く種類。 */
export type IndexEntry = {
  id: string;
  kind: 'part' | 'model' | 'original' | 'replaced' | 'bench';
  name: string;
  created: string;
  /** 計算にかかった秒（run.py が測る。同一性に入らない）。 */
  elapsed?: number;
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
/** 実験（resolved/ の下のディレクトリ）1 つの状態。モデル（とその学習器と学習データ）、系列の元のモデル、組（系列 × モデルの置換系）を名前の順に並べる（宣言の順は API の答えが宣言の写しで並べる）。state は起こした仕事も見た状態。 */
export type ExperimentStatus = {
  name: string;
  /** モデルごとの組み立ての run と、組んだ学習器（群の順）と、学習した入力の元のモデル（学習の系列の original）。state は学習器と学習の入力も見た状態。 */
  models: { model: string; state: State; run: RunStatus; parts: RunStatus[]; train: RunStatus }[];
  series: { series: string; original: RunStatus }[];
  pairs: { series: string; model: string; state: State; run: RunStatus }[];
  benches: { bench: string; state: State; run: RunStatus }[];
};
/** 実験ごとの状態と、起こした仕事。外の画面が status で受け取る。 */
export type Status = { experiments: ExperimentStatus[]; jobs: JobState[] };
