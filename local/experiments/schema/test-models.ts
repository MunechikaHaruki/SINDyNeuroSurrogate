// テスト専用のモデルの宣言。学習器の基準（config/learner.json）に差分を重ね、群分け grouping で並べる（base.ts の partOf と modelFrom）。
// どのテストにも最後に TEST_COMMON（学習の電流を 180 ms に縮める）を当てる。名前は _test_ で始め、どの実験にも入れない。tests/ と resolve.test.ts が名前で引く。

/** どのテストのモデルにも最後に当てる差分。学習の電流を 180 ms に縮める。 */
export const TEST_COMMON = { data: { current_params: { duration: 180 } } };

/** HH の既定の項ライブラリ（細胞のゲートの項と定数・電位の項）。 */
const HH_LIBRARY = [
  { type: 'hh_gate_forward' },
  { type: 'hh_gate_backward' },
  { type: 'hh_gate_forward_product' },
  { type: 'hh_gate_backward_product' },
  { type: 'basis' },
];
/** HH を教師にする学習器の差分。 */
const HH = { cell: 'hh', data: { target: 'hh' } };

/** テスト専用のモデル。grouping は群分け、ほかは学習器の基準への差分（JSON Merge Patch）。 */
export const TEST_DECLARATIONS: Record<string, { grouping: string } & Record<string, unknown>> = {
  /** テストは形と有限性だけを見るので、決定的で速い PCA にする（_test_ はどれもテスト専用で、tests が名前で引く）。 */
  _test_hh_sindy: {
    grouping: 'm+h+n',
    ...HH,
    form: 'sindy',
    group: { d: 3, rhs_params: { library_specs: HH_LIBRARY } },
  },
  _test_hh_hybrid: {
    grouping: 'm+h+n',
    ...HH,
    group: { d: 3, rhs_params: { library_specs: HH_LIBRARY } },
  },
  /** AE の経路を踏むだけなので epochs を切り詰める。 */
  _test_hh_ae: {
    grouping: 'm+h+n',
    ...HH,
    group: { compressor: 'ae', d: 2, compressor_params: { epochs: 20 }, rhs_params: { library_specs: HH_LIBRARY } },
  },
  _test_traub_hybrid: { grouping: 'm+n+c+a+h+b+s+r', group: { d: 5 } },
  _test_traub_hybrid_sr_physics: {
    grouping: 'm+n+c+a+h+b',
    group: {
      d: 5,
      rhs_params: {
        library_specs: [
          { type: 'traub_gate_forward', gates: 'msncahrb' },
          { type: 'traub_gate_backward', gates: 'msncahrb' },
          { type: 'traub_gate_forward_product', gates: 'msncahrb' },
          { type: 'traub_gate_backward_product', gates: 'msncahrb' },
          { type: 'basis' },
        ],
      },
    },
  },
  _test_traub_hybrid_hn: { grouping: 'n+h' },
  /** 群を 2 つ持つ形。{h,n} と {a,c} をそれぞれ 1 本へ縮め、残りは元の式。 */
  _test_traub_two_groups: { grouping: 'n+h_a+c' },
  _test_traub_kam: {
    grouping: 'n+h_a+c',
    group: { compressor: 'kam', rhs: 'kam', rhs_params: { library_specs: null } },
  },
  /** encoder・decoder と潜在方程式（NN）を同じロスで学習する経路。項ライブラリは持たない。 */
  _test_traub_ude: {
    grouping: 'm+n+c+a+h+b+s+r',
    data: { target: 'traub19' },
    group: {
      compressor: 'ae',
      d: 3,
      rhs: 'ude',
      compressor_params: { epochs: 20 },
      rhs_params: {
        library_specs: null,
        epochs: 20,
        lr: 0.003,
        window: 20,
        batch: 16,
        hidden: 32,
        depth: 2,
        w_recon: 1.0,
        pull: 20.0,
      },
    },
  },
};
