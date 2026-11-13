// テスト専用の学習の宣言。仕方（config/presets.json の methods）に差分を重ね、presets.ts の methodOf と trainFrom が学習 run にする。
// 名前は _test_ で始め、どの実験にも入れない。tests/ と resolve.test.ts が名前で引く。

/** テスト専用の学習。仕方に差分を重ね、電流は 180 ms に縮める。名前は _test_ で始め、どの実験にも入れない。tests/ が名前で引く。 */
export const TEST_DECLARATIONS: Record<string, Record<string, unknown>> = {
  /** テストは形と有限性だけを見るので、決定的で速い PCA にする（_test_ はどれもテスト専用で、tests が名前で引く）。 */
  _test_hh_sindy: {
    method: 'pca-sindy',
    grouping: 'm+h+n',
    cell: 'hh',
    form: 'sindy',
    d: 3,
  },
  _test_hh_hybrid: {
    method: 'pca-sindy',
    grouping: 'm+h+n',
    cell: 'hh',
    d: 3,
  },
  /** AE の経路を踏むだけなので epochs を切り詰める。 */
  _test_hh_ae: {
    method: 'ae-sindy',
    grouping: 'm+h+n',
    cell: 'hh',
    d: 2,
    compressor_params: {
      epochs: 20,
    },
  },
  _test_traub_hybrid: {
    method: 'pca-sindy',
    grouping: 'm+n+c+a+h+b+s+r',
    d: 5,
  },
  _test_traub_hybrid_sr_physics: {
    method: 'pca-sindy',
    grouping: 'm+n+c+a+h+b',
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
    method: 'pca-sindy',
    grouping: 'n+h',
  },
  /** 群を 2 つ持つ形。{h,n} と {a,c} をそれぞれ 1 本へ縮め、残りは元の式。 */
  _test_traub_two_groups: {
    method: 'pca-sindy',
    grouping: 'n+h_a+c',
  },
  _test_traub_kam: {
    method: 'kam',
    grouping: 'n+h_a+c',
  },
  /** encoder・decoder と潜在方程式（NN）を同じロスで学習する経路。項ライブラリは持たない。 */
  _test_traub_ude: {
    method: 'ae-sindy',
    grouping: 'm+n+c+a+h+b+s+r',
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
