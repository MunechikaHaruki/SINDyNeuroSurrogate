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
export type LibrarySpec = { type: string };

/**
 * 学習の設定ツリー（Python の SurrogateSpec.from_config が解く）。spec は何を学習するか、
 * preprocessor・ansatz・closure は parts/ の 3 構成要素と同名で、それぞれその層の入口の署名へ ** で展開される。
 * 受理するキーと既定値はその署名だけが決め、綴り違いや層違いは Python で TypeError になる。
 */
export type SurrogateConfig = {
  spec: {
    comp_type: string;
    surrogate_type: string;
    preprocessor_type: string;
    n_components: number;
    datasets: SimSpec;
    /** 学習を 1 ノードへ絞る指定。null は学習ネット内の学習ドメイン全部。 */
    train_comp_identifier: string | null;
  };
  preprocessor: Record<string, unknown>;
  /** hybrid と ude は physics_type（neurons の HYBRID_SPLITS のキー）が必須。sindy は分割が無いので空。 */
  ansatz: { physics_type?: string };
  closure: { library_specs?: LibrarySpec[]; [key: string]: unknown };
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
export const FORMAT = 1;

/** run.py に渡す spec.json。ハッシュを取ると run のディレクトリ名になる。 */
export type Spec =
  | { format: number; kind: 'train'; surrogate: SurrogateConfig }
  | { format: number; kind: 'original'; series: Omit<Series, 'replace_targets'> }
  | { format: number; kind: 'replaced'; series: Series; original: string; train: string };

/** run の一覧の 1 行（runs/index.json）。run の meta.json に spec とハッシュを足したもの。 */
export type IndexEntry = { hash: string; kind: Spec['kind']; name: string; created: string; unit?: string; spec: Spec };

/** run 1 本、または組がいまどこまで回っているか。unreplaceable は置換できないと分かった組。 */
export type State = 'done' | 'running' | 'todo' | 'unreplaceable';
/** 操作盤のボタンが起こす仕事。 */
export type Job = { action: 'train'; preset: string } | { action: 'eval'; series: string; preset: string };
/** 操作盤から起こした仕事 1 つ。failed は終了コードが 0 でないか、終わりを書かずに消えたもの。log はログの末尾の行。 */
export type JobState = { job: Job; started: string; state: 'running' | 'done' | 'failed'; log: string[] };

/** 宣言した run 1 本（学習なら掃引の 1 点）の名前・ハッシュ・状態。図はこのハッシュで runs/ を読む。 */
export type RunStatus = { name: string; hash: string; state: State };
/** 実験 1 つの状態。preset の runs は掃引の点の順、組の runs はその preset の runs と同じ順。 */
export type ExperimentStatus = {
  name: string;
  presets: { preset: string; state: State; runs: RunStatus[] }[];
  series: { series: string; original: RunStatus }[];
  pairs: { series: string; preset: string; state: State; runs: RunStatus[] }[];
};
/** 実験ごとの状態と、起こした仕事。操作盤の口（api.ts）が返す。 */
export type Status = { experiments: ExperimentStatus[]; jobs: JobState[] };
