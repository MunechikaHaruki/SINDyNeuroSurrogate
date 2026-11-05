// 回せるものを組む部品。値はすべて repo の根の presets.json（既定 defaults、縮約の仕方 methods、評価の系列 series）が持ち、ここは値を持たない。
// 実験（experiments/<主題>/schema.ts）は仕方と系列をプリセットの名前か、プリセットからの差分 `{ from, ...差分 }` で指し、ここの関数で学習 run と系列にする。
// 混ぜる順は defaults → プリセット → 差分で、書かなかった欄だけ値から導く（教師は細胞、圧縮は仕方の名前、右辺は圧縮から、項ライブラリは細胞から）。
// テスト専用の学習は presets.json に入れず、隣の test-trains.ts が持つ。形は spec.ts。綴りの確かめはここと、pytest が Python のドメインに当てて見る（tests/test_surrogate.py）。
import { z } from 'zod';
import declaration from '../../presets.json' with { type: 'json' };
import { TEST_DECLARATIONS } from './test-trains.ts';
import type { Group, LibrarySpec, Series, SimSpec, SurrogateConfig, Train } from './spec.ts';

type Entry = Record<string, unknown>;
const raw = declaration as unknown as {
  defaults: { method: Entry & { dt: number }; series: Entry & { dt: number } };
  methods: Record<string, Entry>;
  series: Record<string, Entry>;
};

/** 仕方の値の形。既定値は Python の署名が持つので、書くのは仕方が決める値だけ。 */
const Method = z
  .strictObject({
    cell: z.string(),
    form: z.enum(['hybrid', 'sindy']),
    /** 教師データを取るモデルと、学習用の電流の長さ（ms）。 */
    target: z.string(),
    duration: z.number(),
    compressor: z.enum(['pca', 'ae', 'kam']),
    /** 群の潜在次元（全群で同じ）。 */
    d: z.number(),
    rhs: z.enum(['sindy', 'ude', 'kam']),
    library_specs: z.array(z.looseObject({ type: z.string(), gates: z.string().optional() })).optional(),
    compressor_params: z.record(z.string(), z.unknown()).optional(),
    rhs_params: z.record(z.string(), z.unknown()).optional(),
    /** 膜電位へ吸収するゲート（大文字の状態の名前、`["M"]`）。 */
    absorb: z.array(z.string()).nonempty().optional(),
  })
  .refine(m => m.absorb === undefined || m.form === 'hybrid', 'absorb は hybrid だけ');
export type Declared = z.infer<typeof Method>;

/** 仕方か系列の指し方。プリセットの名前か、プリセットからの差分。 */
export const Ref = z.union([z.string(), z.looseObject({ from: z.string() })]);
export type Ref = z.infer<typeof Ref>;

/** 形の誤りを、どこの何かが分かる一文にする。 */
function parsed<T>(schema: z.ZodType<T>, where: string, value: unknown): T {
  const result = schema.safeParse(value);
  if (!result.success) throw new Error(`${where}：${z.prettifyError(result.error)}`);
  return result.data;
}

/** 指し方をプリセットの名前と差分に分ける。 */
function split(ref: Ref, presets: Record<string, Entry>, kind: string): { from: string; diff: Entry } {
  const { from, ...diff } = typeof ref === 'string' ? { from: ref } : ref;
  if (!presets[from]) throw new Error(`${kind} ${from} が presets.json に無い（${Object.keys(presets).join(', ')}）`);
  return { from, diff };
}

/** 細胞のゲートの項（レートとその積）に定数と電位の項を足した、既定の項ライブラリ。 */
function libraryOf(cell: string): LibrarySpec[] {
  const gates = ['gate_forward', 'gate_backward', 'gate_forward_product', 'gate_backward_product'];
  return [...gates.map(g => ({ type: `${cell}_${g}` })), { type: 'basis' }];
}

/** 仕方を値にする。defaults → プリセット → 差分の順に重ね、書かなかった欄を導く。 */
export function methodOf(ref: Ref): Declared {
  const { from, diff } = split(ref, raw.methods, '仕方');
  const { dt: _dt, ...defaults } = raw.defaults.method;
  const entry: Entry = { ...defaults, ...raw.methods[from], ...diff };
  const cell = String(entry.cell);
  const compressor = entry.compressor ?? from;
  const rhs = entry.rhs ?? (compressor === 'kam' ? 'kam' : 'sindy');
  const library = rhs === 'sindy' ? { library_specs: entry.library_specs ?? libraryOf(cell) } : {};
  return parsed(Method, `仕方 ${JSON.stringify(ref)}`, {
    ...entry,
    target: entry.target ?? cell,
    compressor,
    rhs,
    ...library,
  });
}

/** 群分けの名前（`hn_ac`）を状態の名前の群（[['H','N'],['A','C']]）にする。 */
export function groupsOf(grouping: string): string[][] {
  const groups: string[][] = [];
  for (const block of grouping.split('_')) {
    if (block.length < 2) throw new Error(`群分け ${grouping} の群 ${block} は 2 つ以上のゲートを持たない`);
    groups.push([...block.toUpperCase()]);
  }
  return groups;
}

/** 仕方と群分けを学習 run にする。学習用の電流（train）・刻み・全区画で学習するのは、どの学習でも同じ。name は表示の名前で、同一性には入らない。 */
export function trainFrom(name: string, d: Declared, grouping: string): Train {
  const data: SimSpec = {
    target: d.target,
    current_type: 'train',
    dt: raw.defaults.method.dt,
    current_params: { duration: d.duration },
  };
  const shared = groupsOf(grouping)
    .flat()
    .filter(g => d.absorb?.includes(g));
  if (shared.length > 0) throw new Error(`${name} は ${shared.join('・')} を群と吸収の両方に入れている`);
  const groups: Group[] = groupsOf(grouping).map(gates => ({
    gates,
    compressor: d.compressor,
    d: d.d,
    rhs: d.rhs,
    compressor_params: d.compressor_params ?? {},
    rhs_params:
      d.rhs === 'sindy' && d.library_specs !== undefined ? { library_specs: d.library_specs } : (d.rhs_params ?? {}),
  }));
  const config: SurrogateConfig = {
    cell: d.cell,
    form: d.form,
    data,
    groups,
    train_comp: null,
    ...(d.absorb === undefined ? {} : { absorbed: d.absorb }),
  };
  return { name, config };
}

/** 系列の鍵（ほかの鍵は電流のパラメータ）。 */
const SERIES_KEYS = ['target', 'current', 'stim', 'replace_targets', 'sweep', 'dt'];
const SeriesShape = z.looseObject({
  target: z.string(),
  current: z.string(),
  stim: z.string(),
  replace_targets: z.array(z.string()).nonempty(),
  dt: z.number(),
  /** { 掃引する電流のパラメータ: 値の配列 } の 1 組。 */
  sweep: z.record(z.string(), z.array(z.number())).refine(s => Object.keys(s).length === 1, 'sweep は 1 組'),
});

/** 系列を掃引にする。defaults → プリセット → 差分の順に重ね、既定にない鍵は電流のパラメータにする。 */
export function seriesFrom(ref: Ref): Series {
  const { from, diff } = split(ref, raw.series, '系列');
  const entry = parsed(SeriesShape, `系列 ${JSON.stringify(ref)}`, {
    ...raw.defaults.series,
    ...raw.series[from],
    ...diff,
  });
  const current: Entry = {};
  for (const key of Object.keys(entry)) if (!SERIES_KEYS.includes(key)) current[key] = entry[key];
  const [[param, values]] = Object.entries(entry.sweep);
  const spec: SimSpec = {
    target: entry.target,
    current_type: entry.current,
    dt: entry.dt,
    current_params: current,
    stim: entry.stim,
  } as SimSpec;
  return { spec, replace_targets: entry.replace_targets, param, values };
}

/** プリセットの系列。名前 → 掃引。手元の CLI で名前から回すときに引く。 */
export const PRESET_SERIES: Record<string, Series> = {};
for (const name of Object.keys(raw.series)) PRESET_SERIES[name] = seriesFrom(name);
/** プリセットの仕方の名前。 */
export const METHOD_NAMES = Object.keys(raw.methods);

/** テスト専用の学習 run。仕方に差分を重ね、電流は 180 ms に縮める。 */
export const TESTS: Record<string, Train> = {};
for (const [name, { method, grouping, ...rest }] of Object.entries(TEST_DECLARATIONS))
  TESTS[name] = trainFrom(name, methodOf({ from: String(method), duration: 180, ...rest }), String(grouping));
