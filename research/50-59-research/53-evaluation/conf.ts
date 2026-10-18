// この研究で回す条件の宣言（学習の preset PRESETS と評価の系列 SERIES）と、run の spec.json の形。
// 段取り（orchestrate.ts）がここから spec.json を組み、Python の計算の入口（neurosurrogate/run.py）へ渡す。図（runs.ts）は同じ形で読む。
// node や bun に依存しない（ブラウザで動く runs.ts も import する）。
// 綴りの確かめは型と、pytest が Python のドメインに当てて見る（tests/test_surrogate.py）。
// 描き方（比較する comp、指標、点）はここに持たない。図を見て決め直すもので、research/ の図が持つ。

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
type LibrarySpec = { type: string };

/**
 * 学習の設定ツリー（Python の SurrogateSpec.from_config が解く）。spec は何を学習するか、
 * preprocessor・ansatz・closure は parts/ の 3 構成要素と同名で、それぞれその層の入口の署名へ ** で展開される
 * （preprocessor → fit_pca / fit_ae、ansatz → HybridAnsatz.split、closure → 閉包項の同定入口）。
 * 受理するキーと既定値はその署名だけが決め、綴り違いや層違いは Python で TypeError になる。
 */
export type SurrogateConfig = {
  spec: {
    /** 置換対象のコンパートメント種類 */
    comp_type: string;
    surrogate_type: string;
    preprocessor_type: string;
    /** 潜在方程式の次元 */
    n_components: number;
    datasets: SimSpec;
    /** 学習を 1 ノードへ絞る指定。null は学習ネット内の学習ドメイン全部。 */
    train_comp_identifier: string | null;
  };
  /** 種別固有（ae: epochs/lr。pca は無し） */
  preprocessor: Record<string, unknown>;
  /**
   * 定式化固有。hybrid 系は physics_type（学習と physics の分割の名前 = neurons の HYBRID_SPLITS のキー）が必須で既定を持たない。
   * comp_type 名で代替すると分割を決める源が 2 つになる。sindy は分割そのものが無いので空。
   */
  ansatz: { physics_type?: string };
  /** 閉包項の同定固有（sindy: library_specs/optimizer。ude: joint 学習の epochs/window…） */
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

/** spec の形式の版。neurosurrogate/run.py の _FORMAT と揃える。spec に入るので、変えると鍵が変わり古い run は回し直される。 */
export const FORMAT = 1;

/** run.py に渡す spec.json。ハッシュを取ると run のディレクトリ名になる。 */
export type Spec =
  | { format: number; kind: 'train'; surrogate: SurrogateConfig }
  | { format: number; kind: 'original'; series: Omit<Series, 'replace_targets'> }
  | { format: number; kind: 'replaced'; series: Series; original: string; train: string };

/** run の一覧の 1 行（runs/index.json）。run の meta.json に spec とハッシュを足したもの。 */
export type IndexEntry = { hash: string; kind: Spec['kind']; name: string; created: string; unit?: string; spec: Spec };

/** 宣言の 1 本がいまどこまで回っているか。unreplaceable は置換できないと分かった組。 */
export type State = 'done' | 'running' | 'todo' | 'unreplaceable';
/** 宣言の学習（preset ごと）と評価（preset × 系列ごと）の状態。図のボタンの口（api.ts）が返す。 */
export type Status = {
  trains: { preset: string; state: State }[];
  pairs: { series: string; preset: string; state: State }[];
};
/** 図のボタンが起こす仕事。 */
export type Job = { action: 'train'; preset: string } | { action: 'eval'; series: string; preset: string };

/** 学習 run 1 本の宣言。name は図に出す名前で、同一性には入らない。 */
export type Train = { name: string; config: SurrogateConfig };

type DeepPartial<T> = T extends unknown[] ? T : T extends object ? { [K in keyof T]?: DeepPartial<T[K]> } : T;

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** 深いマージ。入れ子の object は混ぜ、配列と値は patch で置き換える（Hydra の合成と同じ）。 */
export function merge<T>(base: T, patch: DeepPartial<T>): T {
  if (!isObject(base) || !isObject(patch)) return patch as T;
  const out: Record<string, unknown> = { ...base };
  for (const [key, value] of Object.entries(patch)) out[key] = key in out ? merge(out[key], value as never) : value;
  return out as T;
}

/** 掃引する軸の値を 1 つずつ入れた run の列（直積）。名前は preset 名に軸と値を足す。 */
function sweep(name: string, base: SurrogateConfig, axes: { n_components?: number[]; preprocessor_type?: string[] }) {
  const runs: Train[] = [];
  for (const n of axes.n_components ?? [base.spec.n_components])
    for (const p of axes.preprocessor_type ?? [base.spec.preprocessor_type]) {
      const label = [name];
      if (axes.n_components) label.push(`n_components=${n}`);
      if (axes.preprocessor_type) label.push(`preprocessor_type=${p}`);
      runs.push({ name: label.join(' '), config: merge(base, { spec: { n_components: n, preprocessor_type: p } }) });
    }
  return runs;
}

// 3 ブロックとも既定値は実装側の署名が単一源。ここに既定を書くと、その層の実装が違う preset へも黙って継承される。
// 書くのは preset が動かす値だけ。
const BASE: SurrogateConfig = {
  spec: {
    comp_type: '',
    surrogate_type: '',
    preprocessor_type: '',
    n_components: 0,
    datasets: { target: '', current_type: 'train', dt: 0.01, current_params: {} },
    train_comp_identifier: null,
  },
  preprocessor: {},
  ansatz: {},
  closure: {},
};

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

// 教師データ = 19-comp MC の全 comp 軌道（どこを置換するかは preset でなく評価系列が決める: Series.replace_targets）。
const TRAUB_MC19 = merge(BASE, {
  spec: {
    comp_type: 'traub',
    surrogate_type: 'hybrid',
    preprocessor_type: 'pca',
    n_components: 6,
    // 19 comp 全部が学習軌道になる（spec.train_comp_ids）→ 1 軌道を短くして総サンプル数を単体モデル（duration=9000）と同程度に揃える。
    datasets: { target: 'traub19', current_params: { duration: 1000 } },
  },
  ansatz: { physics_type: 'traub' },
  closure: { library_specs: TRAUB_GATES },
});

// テスト専用の preset は名前を _test_ で始める（tests は名前を指すだけ）。本番 preset を継承しない（移動で壊れないため）。
// smoke は shape と有限性だけを見るので、電流を 180 ms に縮める（本番は 9000）。
const SHORT = { current_params: { duration: 180 } };

const TEST_HH_SINDY = merge(BASE, {
  spec: {
    comp_type: 'hh',
    surrogate_type: 'sindy',
    // 決定的で速い pca に固定（AE は 1000 epoch でテスト時間が数倍、乱数初期化で fit 品質もブレる）。AE 経路自体は _test_hh_ae が薄く踏む。
    preprocessor_type: 'pca',
    // 式構造と描画の代表。latent が複数の方が列構造が厳しい
    n_components: 3,
    datasets: { target: 'hh', ...SHORT },
  },
  closure: { library_specs: HH_GATES },
});

const TEST_TRAUB_HYBRID = merge(BASE, {
  spec: {
    comp_type: 'traub',
    surrogate_type: 'hybrid',
    // 決定的で速い pca に固定（AE は乱数初期化で品質がブレる）
    preprocessor_type: 'pca',
    n_components: 5,
    datasets: { target: 'traub', ...SHORT },
  },
  ansatz: { physics_type: 'traub' },
  closure: { library_specs: TRAUB_GATES },
});

const TRAUB_SINDY = merge(BASE, {
  spec: {
    comp_type: 'traub',
    surrogate_type: 'sindy',
    preprocessor_type: 'pca',
    n_components: 1,
    datasets: { target: 'traub' },
  },
});

const HH_HYBRID = merge(BASE, {
  spec: {
    comp_type: 'hh',
    surrogate_type: 'hybrid',
    preprocessor_type: 'pca',
    n_components: 2,
    datasets: { target: 'hh' },
  },
  ansatz: { physics_type: 'hh' },
  closure: { library_specs: HH_GATES },
});

/** 学習の preset。名前 → 学習 run の列（掃引しない preset は 1 本）。 */
export const PRESETS: Record<string, Train[]> = {
  hh_hybrid: sweep('hh_hybrid', HH_HYBRID, { n_components: [1, 2], preprocessor_type: ['pca'] }),
  traub_useMC19CompForTeachingData: sweep('traub_useMC19CompForTeachingData', TRAUB_MC19, {
    n_components: [6, 7, 8],
    preprocessor_type: ['pca'],
  }),
  // 対照群: traub_useMC19CompForTeachingData の S,R physics 版。
  poster_traub_pca: [
    {
      name: 'poster_traub_pca',
      config: merge(TRAUB_MC19, {
        spec: { n_components: 5, preprocessor_type: 'pca' },
        ansatz: { physics_type: 'traub_sr_physics' },
      }),
    },
  ],
  poster_traub_ae: [
    {
      name: 'poster_traub_ae',
      config: merge(TRAUB_MC19, {
        spec: { n_components: 5, preprocessor_type: 'ae' },
        ansatz: { physics_type: 'traub_sr_physics' },
      }),
    },
  ],
  // 圧縮ゼロの上限（基準線）。潜在次元 = 学習ゲート数 8（= TRAUB_LEARNED_GATE_NAMES。Ca サブ系 XI/Q は physics 分離済みで学習対象外）を
  // PCA で取る → 8→8 の直交変換は完全可逆で情報損失ゼロ（実測 再構成 mse 1.5e-32）、かつ SINDy のレート関数ライブラリは
  // 真の dg/dt = α(1-g) - βg を厳密に張れる。
  //
  // 「サロゲート化が原理的に可能か」を切り分けるための対照。ここが合わなければ圧縮以前（定式化・積分・移植）が壊れているので、何を圧縮しても無駄。
  //
  // 実測（traub19 全 19 comp 置換、current_type=train duration=500、soma で評価）:
  //   n=8 pca（この preset）  rmse= 3.92  spikes 5->5   形状corr=1.0000  ← ほぼ完全再現
  //   n=5 pca                rmse=10.19  spikes 5->15  形状corr=1.0000
  //   n=3 pca                rmse=23.11  spikes 5->0   形状corr=nan
  // → 骨格（kernel の結線・Ca physics 分離・Euler 積分・19 comp への移植）は正しく、壁は圧縮そのもの。しかも n=5 の再構成 mse は
  //   7.4e-05 と十分小さいのに dynamics は壊れる = 再構成 MSE だけで潜在座標を選ぶのが誤りという UDE の動機そのもの。
  //
  // 教師データを単体 comp にすると圧縮ゼロでも nan（発散）→ 訓練分布 = 評価分布が移植の必須条件。この preset も 19 comp 全部の軌道で学習する。
  poster_noCompression: [
    {
      name: 'poster_noCompression',
      config: merge(TRAUB_MC19, {
        // n_components = 学習ゲート数。これ以上は増やせない（圧縮率 1.0）。8→8 PCA のみが完全可逆（AE は tanh と隠れ層で損失が出る）
        spec: { n_components: 6, preprocessor_type: 'pca', datasets: { current_params: { duration: 1000 } } },
        ansatz: { physics_type: 'traub_sr_physics' },
      }),
    },
  ],
  _traub_informed: [
    {
      name: '_traub_informed',
      config: merge(TRAUB_SINDY, {
        closure: {
          library_specs: [
            { type: 'gate_poly_volt' }, // latent の高次項・V 積（1 次は basis）
            { type: 'traub_gate_forward' },
            { type: 'traub_gate_forward_product' },
            { type: 'traub_gate_backward_product' },
            { type: 'basis' },
          ],
        },
      }),
    },
  ],
  _traub_relaxation: [
    {
      name: '_traub_relaxation',
      config: merge(TRAUB_SINDY, {
        closure: {
          library_specs: [
            { type: 'gate_poly_volt' }, // latent の高次項・V 積（1 次は basis）
            { type: 'basis' },
            // dg0/dt 用（relaxation 形式 dg/dt = α(V) - (α+β)(V)·g）
            { type: 'traub_gate_forward' }, // 駆動項 α(V)
            { type: 'traub_relaxation_decay' }, // 減衰項 (α+β)(V)·g
          ],
        },
      }),
    },
  ],
  _test_hh_sindy: [{ name: '_test_hh_sindy', config: TEST_HH_SINDY }],
  // _test_hh_sindy の hybrid 版
  _test_hh_hybrid: [
    {
      name: '_test_hh_hybrid',
      config: merge(TEST_HH_SINDY, { spec: { surrogate_type: 'hybrid' }, ansatz: { physics_type: 'hh' } }),
    },
  ],
  // pca 固定の他テストが踏まない AE の encode/decode を通す。epochs を切り詰めるので再構成品質は問わない（経路が通ることだけ）。
  _test_hh_ae: [
    {
      name: '_test_hh_ae',
      config: merge(TEST_HH_SINDY, {
        spec: { surrogate_type: 'hybrid', preprocessor_type: 'ae', n_components: 2 },
        preprocessor: { epochs: 20 }, // 収束は見ない
        ansatz: { physics_type: 'hh' },
      }),
    },
  ],
  // 単体 traub 教師。Ca サブ系（XI/Q）の physics 分離と traub19 への移植を踏む経路。
  _test_traub_hybrid: [{ name: '_test_traub_hybrid', config: TEST_TRAUB_HYBRID }],
  // _test_traub_hybrid の S,R physics 版（学習ゲート 8 → 6）。physics 分割を preset で振れること = 状態次元・初期値・置換が両分割で通ることを踏む。
  _test_traub_hybrid_sr_physics: [
    {
      name: '_test_traub_hybrid_sr_physics',
      config: merge(TEST_TRAUB_HYBRID, { ansatz: { physics_type: 'traub_sr_physics' } }),
    },
  ],
  // UDE = encoder/decoder + 潜在方程式を同じロスで同時学習 → Ca physics 分離 → 置換までの経路を踏む。19-comp 教師。
  // closure は joint 学習のつまみ（library_specs は持たない = 潜在方程式が NN なので項ライブラリの語彙が要らない）。
  // smoke なので epochs・window・batch を切り詰める（回って有限値が出ることだけ見る）。
  _test_traub_ude: [
    {
      name: '_test_traub_ude',
      config: merge(BASE, {
        spec: {
          comp_type: 'traub',
          surrogate_type: 'ude',
          preprocessor_type: 'ae', // ude は encoder/decoder を学習変数にするので ae 固定
          n_components: 3,
          datasets: { target: 'traub19', ...SHORT }, // 本番は 1000
        },
        preprocessor: { epochs: 20 }, // AE は UDE の初期値でしかない（joint 学習が続きを担う）
        ansatz: { physics_type: 'traub' },
        closure: {
          epochs: 20,
          lr: 3.0e-3,
          window: 20, // 短縮電流でも窓が取れる長さ
          batch: 16,
          hidden: 32,
          depth: 2,
          w_recon: 1.0, // 再構成アンカーの重み
          pull: 20.0, // 潜在が値域 [-1,1] を出たときの復元力（closure/ude.py）
        },
      }),
    },
  ],
};

// 掃引つき評価の共通電流パラメータ（刺激前の静穏 + 本体長）。掃引軸の値は入らない（Python の EvalSeries が点ごとに埋める）。
const STIM = { silence_duration: 10.0, duration: 300.0 };
const DT = 0.01;
// traub19 の全 comp（soma は index 8）。形態から導かず綴る。実験がどこを置換すると宣言したかは、適用先の変化に黙って追随してはいけない。
const TRAUB19_ALL = [
  ...['c00', 'c01', 'c02', 'c03', 'c04', 'c05', 'c06', 'c07'],
  'soma',
  ...['c09', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18'],
];
/** 0 から 10 を 5 点 */
const VALUES = [0, 2.5, 5, 7.5, 10];

// 刺激部位だけを変えた対照ペア（soma / dend）。同じ電流軸で比べる。
const SOMASTIM: SimSpec = { target: 'traub19', current_type: 'lin&steady', dt: DT, current_params: STIM, stim: 'soma' };
// dend 刺激の注入先 c09 は soma 隣の active な apical proximal dendrite で、soma へ確実に電流が伝播する（g_Na=15）。
const DENDSTIM: SimSpec = { ...SOMASTIM, stim: 'c09' };

/** 評価の系列。名前 → 掃引。 */
export const SERIES: Record<string, Series> = {
  // 単体 traub の素の応答（置換の足場が動くかを最短で見る）。掃引なしで完結。
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
  // 入力の速さに対する追従（パルス周波数掃引）。
  traub19_pulse_freq: {
    spec: {
      target: 'traub19',
      current_type: 'periodic&pulse',
      dt: DT,
      current_params: { ...STIM, amplitude: 20, baseline: 0.0 },
      stim: 'soma',
    },
    replace_targets: ['soma'],
    param: 'frequency',
    values: [10, 20, 30, 40, 50],
  },
  // somastim と置換範囲だけが違う対照: soma 1 つでなく 19 comp 全部を置換する。19 comp 教師の preset
  // （traub_useMC19CompForTeachingData / poster_*）が本来主張しているのはこちら = 教師データの範囲と置換範囲を揃えた条件。
  traub19_somastim_allcomp: { spec: SOMASTIM, replace_targets: TRAUB19_ALL, param: 'value', values: VALUES },
};
