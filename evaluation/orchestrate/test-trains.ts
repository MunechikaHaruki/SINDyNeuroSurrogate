// テスト専用の学習の宣言。仕方（presets.json の methods）に差分を重ね、conf.ts が既定を埋めて学習 run にする。
// 名前は _test_ で始め、どの実験にも入れない。tests/ と orchestrate.test.ts が名前で引く。

/** テスト専用の学習。仕方に差分を重ね、電流は 180 ms に縮める。名前は _test_ で始め、どの実験にも入れない。tests/ が名前で引く。 */
export const TEST_DECLARATIONS: Record<string, Record<string, unknown>> = {
  /** テストは形と有限性だけを見るので、決定的で速い PCA にする（_test_ はどれもテスト専用で、tests が名前で引く）。 */
  _test_hh_sindy: {
    method: 'pca',
    grouping: 'mhn',
    cell: 'hh',
    form: 'sindy',
    d: 3,
  },
  _test_hh_hybrid: {
    method: 'pca',
    grouping: 'mhn',
    cell: 'hh',
    d: 3,
  },
  /** AE の経路を踏むだけなので epochs を切り詰める。 */
  _test_hh_ae: {
    method: 'ae',
    grouping: 'mhn',
    cell: 'hh',
    d: 2,
    compressor_params: {
      epochs: 20,
    },
  },
  _test_traub_hybrid: {
    method: 'pca',
    grouping: 'mncahbsr',
    d: 5,
  },
  _test_traub_hybrid_sr_physics: {
    method: 'pca',
    grouping: 'mncahb',
    d: 5,
    library_specs: [
      {
        type: 'traub_gate_forward',
        gates: 'msncahrb',
      },
      {
        type: 'traub_gate_backward',
        gates: 'msncahrb',
      },
      {
        type: 'traub_gate_forward_product',
        gates: 'msncahrb',
      },
      {
        type: 'traub_gate_backward_product',
        gates: 'msncahrb',
      },
      {
        type: 'basis',
      },
    ],
  },
  _test_traub_hybrid_hn: {
    method: 'pca',
    grouping: 'nh',
  },
  /** 群を 2 つ持つ形。{h,n} と {a,c} をそれぞれ 1 本へ縮め、残りは元の式。 */
  _test_traub_two_groups: {
    method: 'pca',
    grouping: 'nh_ac',
  },
  _test_traub_kam: {
    method: 'kam',
    grouping: 'nh_ac',
  },
  /** encoder・decoder と潜在方程式（NN）を同じロスで学習する経路。項ライブラリは持たない。 */
  _test_traub_ude: {
    method: 'ae',
    grouping: 'mncahbsr',
    target: 'traub19',
    d: 3,
    rhs: 'ude',
    compressor_params: {
      epochs: 20,
    },
    rhs_params: {
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
};
