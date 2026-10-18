// この研究で何を回すかの宣言。学習の preset（PRESETS）、評価の系列（SERIES）、見比べる組（EXPERIMENTS）。
// 段取り（orchestrate.ts）がここから spec.json を組み、Python の計算の入口（neurosurrogate/run.py）へ渡す。形は spec.ts。
// 綴りの確かめは型と、pytest が Python のドメインに当てて見る（tests/test_surrogate.py）。描き方は持たない。
import type { Experiment, LibrarySpec, Series, SimSpec, SurrogateConfig, Train } from './spec.ts';

/** preset が書く値。53.03 の設定の表に出る鍵と同じ。既定値は Python の署名が持つので、書くのは preset が決める値だけ。 */
type Declared = {
  comp_type: string;
  surrogate_type: string;
  preprocessor_type: string;
  /** 配列なら値ごとに学習 run を 1 本ずつ作る（掃引）。 */
  n_components: number | number[];
  /** 教師データを取るモデルと、学習用の電流の長さ（ms）。 */
  target: string;
  duration: number;
  physics_type?: string;
  library_specs?: LibrarySpec[];
  preprocessor?: Record<string, unknown>;
  closure?: Record<string, unknown>;
};

/** 宣言を学習 run の列にする。学習用の電流（train）・刻み・全区画で学習するのは、どの preset でも同じ。 */
function expand(name: string, d: Declared): Train[] {
  const counts = typeof d.n_components === 'number' ? [d.n_components] : d.n_components;
  const runs: Train[] = [];
  for (const n of counts) {
    const datasets: SimSpec = {
      target: d.target,
      current_type: 'train',
      dt: 0.01,
      current_params: { duration: d.duration },
    };
    const config: SurrogateConfig = {
      spec: {
        comp_type: d.comp_type,
        surrogate_type: d.surrogate_type,
        preprocessor_type: d.preprocessor_type,
        n_components: n,
        datasets,
        train_comp_identifier: null,
      },
      preprocessor: d.preprocessor ?? {},
      ansatz: d.physics_type === undefined ? {} : { physics_type: d.physics_type },
      closure: d.library_specs === undefined ? (d.closure ?? {}) : { library_specs: d.library_specs },
    };
    runs.push({ name: counts.length > 1 ? `${name} n_components=${n}` : name, config });
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

// 19 区画の全軌道で学習する。どこを置換するかは系列（replace_targets）が決める。
// 1 軌道を 1000 ms にして、総サンプル数を単一区画（9000 ms）と同じくらいにする。
const TRAUB19 = {
  comp_type: 'traub',
  surrogate_type: 'hybrid',
  target: 'traub19',
  duration: 1000,
  library_specs: TRAUB_GATES,
};
// テストは形と有限性だけを見るので電流を 180 ms に縮め、決定的で速い PCA にする。
const TEST_HH = { comp_type: 'hh', preprocessor_type: 'pca', target: 'hh', duration: 180, library_specs: HH_GATES };
const TEST_TRAUB = {
  comp_type: 'traub',
  surrogate_type: 'hybrid',
  preprocessor_type: 'pca',
  n_components: 5,
  target: 'traub',
  duration: 180,
  library_specs: TRAUB_GATES,
};

const DECLARED: Record<string, Declared> = {
  // Ca 系（χ, q）だけを物理で解き、残る 8 ゲートを潜在へ縮める。
  traub_useMC19CompForTeachingData: {
    ...TRAUB19,
    physics_type: 'traub',
    preprocessor_type: 'pca',
    n_components: [6, 7, 8],
  },
  // s, r も物理で解き、残る 6 ゲートを縮める。noCompression は 6 → 6 の可逆な PCA で、圧縮しないときの基準。
  poster_noCompression: { ...TRAUB19, physics_type: 'traub_sr_physics', preprocessor_type: 'pca', n_components: 6 },
  poster_traub_pca: { ...TRAUB19, physics_type: 'traub_sr_physics', preprocessor_type: 'pca', n_components: 5 },
  poster_traub_ae: { ...TRAUB19, physics_type: 'traub_sr_physics', preprocessor_type: 'ae', n_components: 5 },

  // テスト専用（tests は名前で引く）。どの実験にも入れないので冊には出ない。
  _test_hh_sindy: { ...TEST_HH, surrogate_type: 'sindy', n_components: 3 },
  _test_hh_hybrid: { ...TEST_HH, surrogate_type: 'hybrid', n_components: 3, physics_type: 'hh' },
  // AE の経路を踏むだけなので epochs を切り詰める。
  _test_hh_ae: {
    ...TEST_HH,
    surrogate_type: 'hybrid',
    preprocessor_type: 'ae',
    n_components: 2,
    physics_type: 'hh',
    preprocessor: { epochs: 20 },
  },
  _test_traub_hybrid: { ...TEST_TRAUB, physics_type: 'traub' },
  _test_traub_hybrid_sr_physics: { ...TEST_TRAUB, physics_type: 'traub_sr_physics' },
  // encoder・decoder と潜在方程式（NN）を同じロスで学習する経路。項ライブラリは持たない。
  _test_traub_ude: {
    comp_type: 'traub',
    surrogate_type: 'ude',
    preprocessor_type: 'ae',
    n_components: 3,
    target: 'traub19',
    duration: 180,
    physics_type: 'traub',
    preprocessor: { epochs: 20 },
    closure: { epochs: 20, lr: 3.0e-3, window: 20, batch: 16, hidden: 32, depth: 2, w_recon: 1.0, pull: 20.0 },
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
  latent_dim: {
    presets: ['traub_useMC19CompForTeachingData'],
    series: ['traub19_somastim'],
  },
};
