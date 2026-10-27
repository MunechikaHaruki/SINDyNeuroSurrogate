// 回せるものの宣言を読む。宣言そのものは隣の conf.json（土台 bases、学習の preset、評価の系列 series）が持ち、
// ここは読んで形を確かめ、preset を土台に重ね、掃引を学習 run の列に展開するだけ。段取り（orchestrate.ts）がここから spec.json を組む。形は spec.ts。
// conf.json の description はその宣言の理由で、spec には入れない。綴りの確かめはここと、pytest が Python のドメインに当てて見る（tests/test_surrogate.py）。
import declaration from './conf.json' with { type: 'json' };
import type { Group, LibrarySpec, Series, SimSpec, SurrogateConfig, Train } from './spec.ts';

/** preset が書く値。既定値は Python の署名が持つので、書くのは preset が決める値だけ。 */
type Declared = {
  cell: string;
  form: 'hybrid' | 'sindy';
  /** 教師データを取るモデルと、学習用の電流の長さ（ms）。 */
  target: string;
  duration: number;
  /** 群。区画の状態の名前で書き、群に入らなかった状態は元の式のまま解く。 */
  groups: string[][];
  compressor: 'pca' | 'ae' | 'kam';
  /** 群の潜在次元（全群で同じ）。配列なら値ごとに学習 run を 1 本ずつ作る（掃引）。 */
  d: number | number[];
  rhs?: 'sindy' | 'ude' | 'kam';
  library_specs?: LibrarySpec[];
  compressor_params?: Record<string, unknown>;
  rhs_params?: Record<string, unknown>;
};

type Entry = Record<string, unknown>;
/** preset に書ける鍵（base と description のほか）。 */
const KEYS = [
  'cell',
  'form',
  'target',
  'duration',
  'groups',
  'compressor',
  'd',
  'rhs',
  'library_specs',
  'compressor_params',
  'rhs_params',
];

function fail(where: string, message: string): never {
  throw new Error(`conf.json の ${where}：${message}`);
}
function isObject(value: unknown): value is Entry {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
function isStrings(value: unknown): value is string[] {
  return Array.isArray(value) && value.every(v => typeof v === 'string');
}

/** 土台に preset を重ね、鍵と値の形を確かめる。土台は一段だけで、土台が別の土台を引くことはない。 */
function declared(name: string, preset: Entry, bases: Record<string, Entry>): Declared {
  const where = `presets.${name}`;
  const merged: Entry = {};
  if (preset.base !== undefined) {
    const base = bases[preset.base as string];
    if (!base) fail(where, `土台 ${String(preset.base)} が bases に無い`);
    for (const key of Object.keys(base)) if (key !== 'description') merged[key] = base[key];
  }
  for (const key of Object.keys(preset)) if (key !== 'base' && key !== 'description') merged[key] = preset[key];
  for (const key of Object.keys(merged)) if (!KEYS.includes(key)) fail(where, `知らない鍵 ${key}`);
  if (typeof merged.cell !== 'string' || typeof merged.target !== 'string') fail(where, 'cell と target は文字列');
  if (merged.form !== 'hybrid' && merged.form !== 'sindy') fail(where, 'form は hybrid か sindy');
  if (typeof merged.duration !== 'number') fail(where, 'duration は数');
  if (!Array.isArray(merged.groups) || !merged.groups.every(isStrings)) fail(where, 'groups は状態の名前の配列の配列');
  if (merged.compressor !== 'pca' && merged.compressor !== 'ae' && merged.compressor !== 'kam')
    fail(where, 'compressor は pca・ae・kam');
  const d = merged.d;
  if (typeof d !== 'number' && !(Array.isArray(d) && d.every(v => typeof v === 'number')))
    fail(where, 'd は数か数の配列');
  if (merged.rhs !== undefined && merged.rhs !== 'sindy' && merged.rhs !== 'ude' && merged.rhs !== 'kam')
    fail(where, 'rhs は sindy・ude・kam');
  if (
    merged.library_specs !== undefined &&
    !(Array.isArray(merged.library_specs) && merged.library_specs.every(s => isObject(s) && typeof s.type === 'string'))
  )
    fail(where, 'library_specs は type を持つ物の配列');
  for (const key of ['compressor_params', 'rhs_params'])
    if (merged[key] !== undefined && !isObject(merged[key])) fail(where, `${key} は物`);
  return merged as Declared;
}

/** 宣言を学習 run の列にする。学習用の電流（train）・刻み・全区画で学習するのは、どの preset でも同じ。 */
function expand(name: string, d: Declared): Train[] {
  const dims = typeof d.d === 'number' ? [d.d] : d.d;
  const runs: Train[] = [];
  for (const dim of dims) {
    const data: SimSpec = {
      target: d.target,
      current_type: 'train',
      dt: 0.01,
      current_params: { duration: d.duration },
    };
    const groups: Group[] = [];
    for (const gates of d.groups) {
      const rhsParams = d.library_specs === undefined ? (d.rhs_params ?? {}) : { library_specs: d.library_specs };
      groups.push({
        gates: [...gates],
        compressor: d.compressor,
        d: dim,
        rhs: d.rhs ?? 'sindy',
        compressor_params: d.compressor_params ?? {},
        rhs_params: rhsParams,
      });
    }
    const config: SurrogateConfig = { cell: d.cell, form: d.form, data, groups, train_comp: null };
    runs.push({ name: dims.length > 1 ? `${name} d=${dim}` : name, config });
  }
  return runs;
}

/** 系列の形を確かめ、description を外す。spec はそのまま Python へ渡す。 */
function series(name: string, entry: Entry): Series {
  const where = `series.${name}`;
  if (!isObject(entry.spec)) fail(where, 'spec は物');
  if (!isStrings(entry.replace_targets)) fail(where, 'replace_targets は区画の名前の配列');
  if (entry.param !== null && typeof entry.param !== 'string') fail(where, 'param は文字列か null');
  if (!Array.isArray(entry.values) || !entry.values.every(v => typeof v === 'number')) fail(where, 'values は数の配列');
  return {
    spec: entry.spec as SimSpec,
    replace_targets: entry.replace_targets,
    param: entry.param,
    values: entry.values,
  };
}

const raw = declaration as unknown as {
  bases: Record<string, Entry>;
  presets: Record<string, Entry>;
  series: Record<string, Entry>;
};

/** 学習の preset。名前 → 学習 run の列（掃引しない preset は 1 本）。 */
export const PRESETS: Record<string, Train[]> = {};
for (const name of Object.keys(raw.presets)) PRESETS[name] = expand(name, declared(name, raw.presets[name], raw.bases));

/** 評価の系列。名前 → 掃引。 */
export const SERIES: Record<string, Series> = {};
for (const name of Object.keys(raw.series)) SERIES[name] = series(name, raw.series[name]);
