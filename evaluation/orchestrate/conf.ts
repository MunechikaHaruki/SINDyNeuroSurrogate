// この研究で何を回すかの宣言。学習の preset（PRESETS）、評価の系列（SERIES）、見比べる組（EXPERIMENTS）。
// 段取り（orchestrate.ts）がここから spec.json を組み、Python の計算の入口（neurosurrogate/run.py）へ渡す。形は spec.ts。
// 綴りの確かめは型と、pytest が Python のドメインに当てて見る（tests/test_surrogate.py）。描き方は持たない。
import type { Experiment, Group, LibrarySpec, Series, SimSpec, SurrogateConfig, Train } from './spec.ts';

/** preset が書く値。既定値は Python の署名が持つので、書くのは preset が決める値だけ。 */
type Declared = {
  cell: string;
  form: 'hybrid' | 'sindy';
  /** 教師データを取るモデルと、学習用の電流の長さ（ms）。 */
  target: string;
  duration: number;
  /** 群。区画の状態の名前で書き、群に入らなかった状態は元の式のまま解く。 */
  groups: readonly (readonly string[])[];
  compressor: 'pca' | 'ae';
  /** 群の潜在次元（全群で同じ）。配列なら値ごとに学習 run を 1 本ずつ作る（掃引）。 */
  d: number | number[];
  rhs?: 'sindy' | 'ude';
  library_specs?: LibrarySpec[];
  compressor_params?: Record<string, unknown>;
  rhs_params?: Record<string, unknown>;
};

/** 宣言を学習 run の列にする。学習用の電流（train）・刻み・全区画で学習するのは、どの preset でも同じ。 */
function expand(name: string, d: Declared): Train[] {
  const dims = typeof d.d === 'number' ? [d.d] : d.d;
  const runs: Train[] = [];
  for (const dim of dims) {
    const data: SimSpec = { target: d.target, current_type: 'train', dt: 0.01, current_params: { duration: d.duration } };
    const groups: Group[] = [];
    for (const gates of d.groups) {
      const rhsParams = d.library_specs === undefined ? (d.rhs_params ?? {}) : { library_specs: d.library_specs };
      groups.push({ gates: [...gates], compressor: d.compressor, d: dim, rhs: d.rhs ?? 'sindy', compressor_params: d.compressor_params ?? {}, rhs_params: rhsParams });
    }
    const config: SurrogateConfig = { cell: d.cell, form: d.form, data, groups, train_comp: null };
    runs.push({ name: dims.length > 1 ? `${name} d=${dim}` : name, config });
  }
  return runs;
}

const HH_GATES: LibrarySpec[] = [
  { type: 'hh_gate_forward' },
  { type: 'hh_gate_backward' },
  { type: 'hh_gate_forward_product' },
  { type: 'hh_gate_backward_product' },
  { type: 'basis' },
];
const TRAUB_GATES: LibrarySpec[] = [
  { type: 'traub_gate_forward' },
  { type: 'traub_gate_backward' },
  { type: 'traub_gate_forward_product' },
  { type: 'traub_gate_backward_product' },
  { type: 'basis' },
];

// 群は区画の状態の名前で書く。Traub の状態は M N C A H B S R Q XI、HH は M H N。
const TRAUB_8 = ['M', 'N', 'C', 'A', 'H', 'B', 'S', 'R']; // 電位だけで動く 8 ゲート。Ca 系（Q, XI）は元の式
const TRAUB_6 = ['M', 'N', 'C', 'A', 'H', 'B']; // s, r も元の式で解く
const HH_3 = ['M', 'H', 'N'];

// 群 {h,n} だけを学習する（55.01 の 1 段目）。レート項も h, n のものに絞る。
const HN_GATES: LibrarySpec[] = [];
for (const spec of TRAUB_GATES) HN_GATES.push(spec.type === 'basis' ? spec : { ...spec, gates: 'hn' });
// 単一区画の軌道で学習する。電流は 19 区画の学習（1000 ms × 19）と同じくらいのサンプル数にする。
const TRAUB_HN = { cell: 'traub', form: 'hybrid', target: 'traub', duration: 9000, groups: [['N', 'H']], library_specs: HN_GATES } as const;

// 19 区画の全軌道で学習する。どこを置換するかは系列（replace_targets）が決める。
// 1 軌道を 1000 ms にして、総サンプル数を単一区画（9000 ms）と同じくらいにする。
const TRAUB19 = { cell: 'traub', form: 'hybrid', target: 'traub19', duration: 1000, library_specs: TRAUB_GATES } as const;
// テストは形と有限性だけを見るので電流を 180 ms に縮め、決定的で速い PCA にする。
const TEST_HH = { cell: 'hh', target: 'hh', duration: 180, groups: [HH_3], compressor: 'pca', library_specs: HH_GATES } as const;
const TEST_TRAUB = { cell: 'traub', form: 'hybrid', target: 'traub', duration: 180, compressor: 'pca', d: 5, library_specs: TRAUB_GATES } as const;

const DECLARED: Record<string, Declared> = {
  // Ca 系（χ, q）だけを元の式で解き、残る 8 ゲートを潜在へ縮める。
  traub_useMC19CompForTeachingData: { ...TRAUB19, groups: [TRAUB_8], compressor: 'pca', d: [6, 7, 8] },
  // s, r も元の式で解き、残る 6 ゲートを縮める。noCompression は 6 → 6 の可逆な PCA で、圧縮しないときの基準。
  poster_noCompression: { ...TRAUB19, groups: [TRAUB_6], compressor: 'pca', d: 6 },
  poster_traub_pca: { ...TRAUB19, groups: [TRAUB_6], compressor: 'pca', d: 5 },
  poster_traub_ae: { ...TRAUB19, groups: [TRAUB_6], compressor: 'ae', d: 5 },

  // h, n を 1 本へ縮める。PCA の 2 本は 2 → 2 の可逆な変換で、圧縮しないときの基準。
  grouped_hn_pca: { ...TRAUB_HN, compressor: 'pca', d: [1, 2] },
  grouped_hn_ae: { ...TRAUB_HN, compressor: 'ae', d: 1 },

  // テスト専用（tests は名前で引く）。どの実験にも入れないので冊には出ない。
  _test_hh_sindy: { ...TEST_HH, form: 'sindy', d: 3 },
  _test_hh_hybrid: { ...TEST_HH, form: 'hybrid', d: 3 },
  // AE の経路を踏むだけなので epochs を切り詰める。
  _test_hh_ae: { ...TEST_HH, form: 'hybrid', compressor: 'ae', d: 2, compressor_params: { epochs: 20 } },
  _test_traub_hybrid: { ...TEST_TRAUB, groups: [TRAUB_8] },
  _test_traub_hybrid_sr_physics: { ...TEST_TRAUB, groups: [TRAUB_6] },
  _test_traub_hybrid_hn: { ...TEST_TRAUB, groups: [['N', 'H']], d: 1, library_specs: HN_GATES },
  // 群を 2 つ持つ形。{h,n} と {a,c} をそれぞれ 1 本へ縮め、残りは元の式。
  _test_traub_two_groups: { ...TEST_TRAUB, groups: [['N', 'H'], ['A', 'C']], d: 1 },
  // encoder・decoder と潜在方程式（NN）を同じロスで学習する経路。項ライブラリは持たない。
  _test_traub_ude: {
    cell: 'traub',
    form: 'hybrid',
    target: 'traub19',
    duration: 180,
    groups: [TRAUB_8],
    compressor: 'ae',
    d: 3,
    rhs: 'ude',
    compressor_params: { epochs: 20 },
    rhs_params: { epochs: 20, lr: 3.0e-3, window: 20, batch: 16, hidden: 32, depth: 2, w_recon: 1.0, pull: 20.0 },
  },
};

/** 学習の preset。名前 → 学習 run の列（掃引しない preset は 1 本）。 */
export const PRESETS: Record<string, Train[]> = {};
for (const name of Object.keys(DECLARED)) PRESETS[name] = expand(name, DECLARED[name]);

const DT = 0.01;
/** 刺激前の静穏と本体の長さ（ms）。掃引する値は Python の EvalSeries が点ごとに埋める。 */
const STIM = { silence_duration: 10.0, duration: 300.0 };
// traub19 の全区画（soma は 8 番）。置換する範囲は宣言で綴り、形態から導かない。
const TRAUB19_ALL = [
  ...['c00', 'c01', 'c02', 'c03', 'c04', 'c05', 'c06', 'c07'],
  'soma',
  ...['c09', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18'],
];
/** 0 から 10 µA/cm² を 5 点 */
const VALUES = [0, 2.5, 5, 7.5, 10];
const SOMASTIM: SimSpec = { target: 'traub19', current_type: 'lin&steady', dt: DT, current_params: STIM, stim: 'soma' };
// c09 は soma の隣の apical の近位樹状突起で、電流が soma へ確実に伝わる。
const DENDSTIM: SimSpec = { ...SOMASTIM, stim: 'c09' };

/** 評価の系列。名前 → 掃引。 */
export const SERIES: Record<string, Series> = {
  // 単一区画の素の応答。テストが使う。
  traub_soma_dc: {
    spec: {
      target: 'traub',
      current_type: 'lin&steady',
      dt: DT,
      current_params: { silence_duration: 10.0, duration: 40.0, value: 3.0 },
      stim: 'soma',
    },
    replace_targets: ['soma'],
    param: null,
    values: [],
  },
  // 単一区画を soma への定常電流で掃引する。
  traub_somastim: {
    spec: { target: 'traub', current_type: 'lin&steady', dt: DT, current_params: STIM, stim: 'soma' },
    replace_targets: ['soma'],
    param: 'value',
    values: VALUES,
  },
  traub19_somastim: { spec: SOMASTIM, replace_targets: ['soma'], param: 'value', values: VALUES },
  traub19_dendstim: { spec: DENDSTIM, replace_targets: ['soma'], param: 'value', values: VALUES },
  // somastim と置換範囲だけが違う。教師データの範囲（19 区画）と置換範囲を揃えた条件。
  traub19_somastim_allcomp: { spec: SOMASTIM, replace_targets: TRAUB19_ALL, param: 'value', values: VALUES },
};

/** 見比べる組。冊はここから選ぶ。 */
export const EXPERIMENTS: Record<string, Experiment> = {
  poster: {
    presets: ['poster_noCompression', 'poster_traub_pca', 'poster_traub_ae'],
    series: ['traub19_somastim', 'traub19_dendstim', 'traub19_somastim_allcomp'],
  },
  grouped_hn: {
    presets: ['grouped_hn_pca', 'grouped_hn_ae'],
    series: ['traub_somastim'],
  },
  latent_dim: {
    presets: ['traub_useMC19CompForTeachingData'],
    series: ['traub19_somastim'],
  },
};
