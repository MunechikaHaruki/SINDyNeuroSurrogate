// 帰りの JSON の形。計算する機械の API（remote/serve.ts）の口ごとの受け取りと返事で、手元（冊とエージェント用の CLI）はこの型だけを import して POST /<口> に JSON を送る。
// 口は実験の目次と実験 1 つ、run の置き場（runs/ の inputs・models・data・parts・bench）ごとの 1 つ、run の一覧、仕事を起こす口。
// 計算する機械が run の保存の形を知り、指標の計算も線の間引きも答えの組み立ても済ませて、すぐ描ける形で返す。手元は run のパスも列の位置も知らず、描くことと選ぶことだけを持つ。
// 波形は回線に出さない。返すのは間引いた線・軌跡・分布・指標・表の値だけ。
import type { BenchRow } from './runs.ts';
import type { Job, JobState, State } from './spec.ts';

export type { BenchRow } from './runs.ts';
export type { Job, JobState, State } from './spec.ts';

/** 線の 1 点。 */
export type Point = { x: number; y: number };
/** 線 1 本。 */
export type Line = { label: string; points: Point[] };
/** 時間の範囲 [ms]。省けば系列の全体。 */
export type Window = [number, number];

/** 実験の進み具合。置換できない組は済んだものとして数え、回す run が無ければ未着手。 */
export type Progress = '結果あり' | '計算中' | '途中' | '未着手';
/** run の置き場（runs/ の直下）の名前。目次も画面の URL もこの名前で run を分ける。 */
export type RunKind = 'data' | 'parts' | 'models' | 'inputs' | 'bench';
/** 目次の実験 1 つ。runs は置き場ごとの、実験の run の名前（宣言の順。data・parts は使うモデルの宣言の順）。 */
export type ExperimentEntry = {
  id: string;
  description: string;
  progress: Progress;
  runs: Record<RunKind, string[]>;
};
/** 宣言の名前で引く run 1 本と、その状態。 */
export type NamedRun = { name: string; state: State };
/** 実験の目次。実験群と実験は宣言の順。running は回っている run の数、runs は完成した run の数（増えたら図を読み直す）。measures は比べ方の物差し。 */
export type Experiments = {
  groups: { name: string; description: string; experiments: ExperimentEntry[] }[];
  running: number;
  runs: number;
  measures: { coincidenceWindow: number; gammaLine: number };
};

/** 設定の表の 1 行。label は軸、items は宣言の名前と、その名前が指す値。 */
export type SetupRow = { label: string; items: { name: string; value: string }[] };
/** 実験 1 つ。並びは宣言の順。rerun は実験が使う run の計算にかかった秒の和と、回した数・全部の数。models はモデルごとに組んだ学習データと学習器の名前、pairs は入力 × モデルの評価、benches は GPU の実測。 */
export type ExperimentView = {
  id: string;
  group: string;
  description: string;
  setup: SetupRow[];
  rerun: { seconds: number; done: number; total: number };
  inputs: string[];
  models: { name: string; data: string | null; parts: string[] }[];
  data: NamedRun[];
  parts: NamedRun[];
  pairs: { input: string; model: string; state: State }[];
  benches: NamedRun[];
};

/** 1 本の波形の発火。times は AP の山の時刻 [ms]、features は AP ごとの特徴量（eFEL。times と同じ順）。null は値が無い（nan）。 */
export type SpikeTrain = { times: number[]; features: Record<string, (number | null)[]> };
/** 列 1 本のまとめ。gamma は決まる点での Γ の平均、rheobase は発火の始まり [原系, 置換系]（掃引しなければ null）。 */
export type ColumnSummary = { gamma: number | null; rheobase: [number | null, number | null] };

/** 入力（系列）1 つ。points は掃引の点の見出し（単発なら「単発」）、comps は区画の名前、stimulus は刺激が入る区間 [ms]、duration は系列の長さ [ms]。
 * spec は電流の宣言（表に出す）、original は元のモデルの run の id。 */
export type InputView = {
  name: string;
  param: string | null;
  unit: string;
  values: number[];
  points: string[];
  comps: string[];
  replaceTargets: string[];
  stimulus: Window;
  duration: number;
  dt: number;
  spec: { target: string; current_type: string; stim: string; current_params: Record<string, number> };
  original: { state: State; run: string };
};
/** 入力を 1 つのモデルで置き換えて回し終えた列。reduced は ExperimentView と同じ。 */
export type ColumnView = { model: string; method: string; grouping: string; reduced: number; run: string };
/** 電位の線 1 本。model は列のモデルの名前で、原系は null。 */
export type VoltageLine = Line & { model: string | null };
/** 入力の頼み方。comp は比べる区画（省けば置き換えた区画のうち刺激する区画、無ければ最初に置き換える区画）、models は重ねるモデル（省けば回し終えた全部）、window は線の時間の範囲、waves が false なら線を返さない（指標だけを使う実験のページ）。
 * pair は 1 組（元のモデルと、回し終えたモデル model の列）の掃引の点 point での潜在空間（axes は相平面の軸、V か潜在の名前。sameScale は縦横の単位が同じか）、whole は元のモデルの掃引の点 point の区画 comps の全区画の図。 */
export type InputsRequest = {
  experiment: string;
  input: string;
  comp?: string;
  models?: string[];
  window?: Window;
  waves?: boolean;
  pair?: { model: string; point: number; axes?: { x: string; y: string } };
  whole?: { point: number; comps: string[] };
};
/** 1 組の潜在空間。latents と phase の原系は列の圧縮器で潜在へ写したもの（射影）。比べる区画を置き換えていなければ latents は空。置換系が発散した点では原系だけ。 */
export type PairCompare = {
  model: string;
  latents: Record<string, Line[]>;
  phase: Line[];
  axes: { x: string; y: string; sameScale: boolean };
  /** 原系の軌道の上の潜在の傾きの比べ（列の学習器の右辺と正解）。置き換えていない区画か、右辺を評価しない形（form=sindy）なら null。error は掃引の点の全区間。 */
  slopes: SlopeCompare | null;
};
/** 元のモデルの、区画ごとの膜電位・内部電流・ゲート（0〜1 の状態）・ゲートでない状態（Traub の Ca 濃度 XI など）。 */
export type WholeWaves = { voltage: Line[]; internal: Line[]; gates: Line[]; states: Line[] };
/** 入力の答え。points は掃引の点ごとの注入電流と、原系と発散していない列の電位（縦軸を揃えられるよう全点）と、発散した列。
 * キーはどれもモデルの名前。summaries は列ごとのまとめ、spikes は掃引の点ごとの発火（原系と列）、pair と whole は頼んだときだけの 1 組と全区画の図。 */
export type InputsReply = {
  input: InputView;
  comp: string;
  columns: ColumnView[];
  points: { current: Point[]; voltages: VoltageLine[]; diverged: string[] }[];
  summaries: Record<string, ColumnSummary>;
  spikes: { original: SpikeTrain[]; columns: Record<string, SpikeTrain[]> };
  pair: PairCompare | null;
  whole: WholeWaves | null;
};

/** モデルのページの頼み方。model は宣言の名前（実験の models の一覧）。 */
export type ModelsRequest = { experiment: string; model: string; window?: Window };
/** 組んだ 1 刻みの演算の回数の 1 行。op は演算（exp・div・pm・mul）、diff は置換系 − 元のモデル。 */
export type CostRow = { op: string; surrogate: number; original: number; diff: number };
/** KAM のモデルの群 1 つ。gates は群のゲート、latent は等価電位をまとめた ψ の名前（h+n.z1）。
 * 回し終えていれば、weights は Kepler 1992 の式(6)の重み α_i（gates の順。どれも正であること）、criterion は式(10)の左辺の最大（小さいこと）。どれも区画の型の静止状態で読む。 */
export type KamGroup = { gates: string[]; latent: string; weights: number[] | null; criterion: number | null };
/** KAM のモデルの図（Kepler 1992 の図 2〜4 にあたる）。学習データ data の区画 comp で、時間の範囲 window（系列の長さ duration [ms]）の線。
 * current は区画に入る電流（単一区画では注入電流）、membrane は膜電位と吸収したゲートの等価電位（元のモデル）、potentials は群ごとの、群のゲートの等価電位とそれをまとめた ψ（元のモデル）、
 * voltage は同じ電流での膜電位（元のモデルと縮約）、phase は群ごとの (V, ψ) の軌跡（元のモデルと縮約）。等価電位は v_i = x̄_i⁻¹(x_i) [mV]。 */
export type KamFigures = {
  data: string;
  comp: string;
  duration: number;
  window: Window;
  current: Line[];
  membrane: Line[];
  potentials: { latent: string; lines: Line[] }[];
  voltage: Line[];
  phase: { latent: string; lines: Line[] }[];
};
/** モデル 1 本（モデルの run と、KAM はその spec が指す学習データ）。cell・form・absorbed は組み立て、parts は組んだ学習器の名前（群の順。KAM は空）、cost は組んだ 1 刻みの演算の回数（まだ回していなければ null）。
 * kam は KAM のモデルだけ：groups は群、figures は図（まだ回していなければ null）。 */
export type ModelsReply = {
  model: string;
  run: { state: State; run: string };
  cell: string;
  form: string;
  absorbed: string[];
  parts: string[];
  cost: CostRow[] | null;
  kam: { groups: KamGroup[]; figures: KamFigures | null } | null;
};

/** 学習データの頼み方。data は宣言の名前（実験の data の一覧）、comps は描く区画（省けば刺激した区画）。 */
export type DataRequest = { experiment: string; data: string; comps?: string[]; window?: Window };
/** 学習データ 1 本（run が持つ波形そのもの）。current は注入電流、voltage は区画ごとの膜電位、gates は区画ごとのゲート（0〜1 の状態）、states はゲートでない状態（Traub の Ca 濃度 XI など）、internal は区画ごとの内部電流（区画が 2 つ以上の型だけ。単一区画では注入電流と同じ）。 */
export type DataReply = {
  run: string;
  stim: string;
  /** run が持つ区画と、描いた区画。 */
  comps: string[];
  shown: string[];
  duration: number;
  current: Line[];
  voltage: Line[];
  gates: Line[];
  states: Line[];
  internal: Line[];
};

/** 学習器の頼み方。part は宣言の名前（実験の parts の一覧）、comps は線を描く区画（学習データの上で。省けば学習した最初の区画）、window は線の時間の範囲（省けば全体）。 */
export type PartsRequest = { experiment: string; part: string; comps?: string[]; window?: Window };
/** SINDy の右辺の 1 本（対象 target の時間微分）。項をレートの因子 rate（α_n(V) や (α_n + β_n)(V)。無ければ空）ごとにまとめ、
 * 因子ごとに残りの単項 monomial（1、z_{1}、V など）と係数 xi を項ライブラリの順に並べる。どれも sympy の LaTeX のまま。 */
export type SindyEquation = {
  target: string;
  groups: { rate: string; terms: { monomial: string; xi: number }[] }[];
};
/** 元のモデルの軌道の上の、潜在の傾きの比べ。正解は軌道を潜在へ写した z(t) の前進差分、学習器はその時刻の潜在と状態に右辺が返す dz/dt。
 * lines は潜在ごとの時系列（正解・学習器）、match は潜在ごとの正解 × 学習器の散布（対角に乗れば一致）、error は潜在ごとの全区間の
 * 差の RMSE と正解の標準偏差（RMSE ÷ 標準偏差が、傾きの動きに対するずれの大きさ）。区画が 2 つ以上なら名前に区画を添える。 */
export type SlopeCompare = {
  lines: Record<string, Line[]>;
  match: Line[];
  error: { latent: string; rmse: number; scale: number }[];
};
/** 圧縮器に固有の中身（compressor/own.json）。PCA は寄与率（捨てた成分も）と保持した数と、主成分の向き components（保持した成分 × 群のゲート）と平均 mean（群のゲートごと）、AE は隠れ層の幅。 */
export type CompressorOwn =
  | { kind: 'pca'; ratios: number[]; kept: number; components: number[][]; mean: number[] }
  | { kind: 'ae'; hidden: number };
/** 右辺に固有の中身（rhs/own.json）。SINDy は項ライブラリ、ほかの設定、0 でない係数の数と全係数の数、対象ごとの式。UDE は層の形と学習の損失。 */
export type RhsOwn =
  | { kind: 'sindy'; library: string[]; settings: string[]; nnz: number; size: number; equations: SindyEquation[] }
  | { kind: 'ude'; layers: number[][]; losses: Record<string, number | null> };
/** 学習データの上の中身（fit/<区画>/。学習データから学習した学習器だけ）。線はどれも描いた区画の時間の範囲 window のもの：voltage は膜電位、gates は群のゲート、
 * latent は潜在、recon は再構成誤差、path は潜在の軌跡（軸は axes。sameScale は縦横の単位が同じか）、slopes は傾きの比べ。
 * reconstruction は学習した区画ごとの全区間のまとめ：再構成の MSE と、それをゲートの分散で割ったもの（summary.json）、時刻ごとの RMSE の平均と最大。 */
export type Fit = {
  /** 学習した学習データの名前（実験の data の一覧）。 */
  data: string;
  /** 学習した区画と、描いた区画。 */
  comps: string[];
  shown: string[];
  duration: number;
  window: Window;
  voltage: Line[];
  gates: Line[];
  latent: Line[];
  recon: Line[];
  path: Line[];
  axes: { x: string; y: string; xIsVoltage: boolean; sameScale: boolean };
  slopes: SlopeCompare;
  reconstruction: { comp: string; mse: number | null; ratio: number | null; mean: number; max: number }[];
};
/** 群のゲートの空間（座標は gates の順）。圧縮器がゲートの動きをどの軸へ潰すかを見る。
 * axes は潜在の軸ごとの圧縮の軸：decode の曲線（ほかの軸は学習データの平均）をゲートの空間の点の並びにしたもの（z はその点の潜在の値、z の増える順）。
 * data は学習データの軌道を描いた区画ごとに、時間の範囲で間引いた点。
 * levels は encoder の等高線の格子で、群のゲート 2 本（位置 x < y、gates の順）ずつ、格子 xs × ys の上の潜在ごとの値（ys の行 × xs の列、有限でなければ null）。
 * encoder が非線形な AE だけが持ち、PCA は空（等高線は軸に垂直な直線になる）。 */
export type GateSpace = {
  axes: { latent: string; points: number[][]; z: number[] }[];
  data: { label: string; points: number[][] }[];
  levels: {
    x: number;
    y: number;
    xs: number[];
    ys: number[];
    latents: { latent: string; values: (number | null)[][] }[];
  }[];
};
/** 学習器 1 本。欄は学習器の run の置き場に 1 つずつ対応する：組み立て（spec.json）の label・setup・latents・gates、ゲートの空間（compressor/decode.json と学習データ）、
 * 圧縮器に固有の中身（compressor/own.json）、右辺に固有の中身（rhs/own.json）、学習データの上の中身（fit/）。
 * label は群の名前（h+n）、setup は組み立ての表、gates は群のゲート、gateOrder は区画の型の状態の並び（ゲートの色をほかのページと揃えるのに使う）、space は群のゲートの空間の軌道と圧縮の軸。 */
export type PartsReply = {
  run: string;
  label: string;
  setup: { label: string; value: string }[];
  latents: string[];
  gates: string[];
  gateOrder: string[];
  space: GateSpace;
  compressor: CompressorOwn;
  rhs: RhsOwn;
  fit: Fit;
};

/** GPU の実測の頼み方。files ならツールの出力のファイルそのものも返す。 */
export type BenchRequest = { experiment: string; bench: string; files?: boolean };
/** GPU の実測 1 本。model が null なら元のモデル、dt は入力の刻み [ms]、rows は細胞の数ごとの行（計算する機械がツールの出力から集める）。
 * files は細胞の数 → ファイルの名前 → 中身（無ければ null）、tools はファイルの名前 → それを書いたツール。 */
export type BenchReply = {
  name: string;
  input: string;
  model: string | null;
  cells: number[];
  ncu: boolean;
  dt: number;
  run: { state: State; run: string };
  machine: { device: string; host: string; created: string } | null;
  rows: BenchRow[];
  files?: Record<number, Record<string, string | null>>;
  tools: Record<string, string>;
};

/** run の一覧の 1 行。place は置き場（id の先頭）、name は宣言から引いた名前（どの実験にも当たらなければ null、そのときは回したときの名前 given）。 */
export type RegistryRun = { id: string; place: RunKind; name: string | null; given: string; created: string };
/** run の一覧（新しい順）と、外の画面から起こした仕事。 */
export type RegistryReply = { runs: RegistryRun[]; jobs: JobState[] };

/** 口ごとの受け取りと返事。 */
export type Api = {
  /** 実験の目次（実験群 → 実験と、その run の名前）と、回っている数。 */
  experiments: { body: Record<string, never>; reply: Experiments };
  /** 実験 1 つの設定と、回す組の状態。 */
  experiment: { body: { experiment: string }; reply: ExperimentView };
  /** 入力 1 つを、回し終えたモデルの列と比べたもの（頼めば 1 組の潜在空間と全区画の図も）。 */
  inputs: { body: InputsRequest; reply: InputsReply };
  /** モデル 1 本の組み立てと演算の回数。 */
  models: { body: ModelsRequest; reply: ModelsReply };
  /** 学習データ 1 本。 */
  data: { body: DataRequest; reply: DataReply };
  /** 学習器 1 本。 */
  parts: { body: PartsRequest; reply: PartsReply };
  /** GPU の実測 1 本。 */
  bench: { body: BenchRequest; reply: BenchReply };
  /** run の一覧と仕事。 */
  registry: { body: Record<string, never>; reply: RegistryReply };
  /** 学習か評価を切り離して回す。同じ組が回っていれば起こさない。 */
  launch: { body: { job: Job }; reply: { started: boolean } };
};
