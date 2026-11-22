// 手元の側の出口。宣言（../config/ と隣の TS）から、実験ごとに送る spec.json を組んで resolved/ に置く。テスト専用の学習の設定も resolved/tests.json（計算する機械の pytest が読む）に、宣言の写しを resolved/plans.json（計算する機械の API が画面の答えを組むのに読む）に置く。計算する機械（../../remote/）は宣言を読まず、ここが書いた resolved/ だけを受け取って回す。
// パスは識別のための名前で、送るのは中身だけ：学習データ `<実験>/data/<対象>.<電流>.<パラメータ>.json`、学習器 `<実験>/parts/<圧縮器と右辺>.<群>.json`、それを組んだモデル `<実験>/models/<モデル>.json`、
// `<実験>/sim/<系列>/original.json`（元のモデル）と `<モデル>.json`、`<実験>/bench/<GPU の実測>.json`。宣言のモデル 1 つは、学習データ 1 つ・群ごとの学習器・それを組んだモデルの spec に分かれる。
// KAM のモデル（全部の群が kam）は学習器を持たず、群と学習データ（元のモデルと縮約を描くのに使う）をモデルの spec が直に持つ。KAM と学習器を混ぜたモデルは組まない。
// どちらも名前は中身から付け、同じ中身は実験の中で 1 ファイルにまとめ、同じ名前に違う中身が来たら投げる（名前に出ない違いがある）。
//   bun local/resolve.ts write        resolved/ を今の宣言から書き直す
//   … check                                          resolved/ が今の宣言と同じか（違えば挙げて終了コード 1。local/remote.sh up が送る前に見る）
//   引数なしで名前の一覧
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { MEASURES, METHOD_NAMES, TESTS, type ModelDecl, type Series } from './experiments/schema/presets.ts';
import {
  EXPERIMENTS,
  GROUPS,
  MODELS,
  SERIES,
  modelOf,
  seriesOf,
  type GroupedPlan,
} from './experiments/schema/experiments.ts';
import { hashOf } from '../shared/hash.ts';
import { FORMATS, type Device, type Group, type Input, type Plans, type Spec } from '../shared/spec.ts';
import type { Bench } from './experiments/schema/schema.ts';

/** 送る spec.json の置き場。 */
export const RESOLVED = fileURLToPath(new URL('./experiments/resolved', import.meta.url));

/** 学習器の学習は GPU（AE の学習が速い）、テストの学習は GPU の無い機械でも回るよう CPU。 */
function trainDevice(model: ModelDecl): Device {
  return model.name.startsWith('_test_') ? 'cpu' : 'gpu';
}

/** 学習データの spec。元のモデルを学習用の電流で回すだけなので CPU。 */
export function dataSpec(model: ModelDecl): Spec {
  return { format: FORMATS.data, device: 'cpu', kind: 'data', data: model.config.data };
}

/** KAM の群か。KAM は学習せず、区画の型の式だけから組む。 */
export function isKam(group: Group): boolean {
  return group.compressor === 'kam';
}

/** KAM のモデルか（全部の群が kam）。KAM と学習器を混ぜたモデルは投げる。 */
export function kamModel(model: ModelDecl): boolean {
  const kam = model.config.groups.filter(isKam).length;
  if (kam > 0 && kam < model.config.groups.length) throw new Error(`${model.name} は KAM と学習器を混ぜている`);
  return kam > 0;
}

/** 群 1 つの学習器の spec。学習データをハッシュで指す。KAM の群は学習器を持たない。 */
export function partSpec(model: ModelDecl, group: Group): Spec {
  if (isKam(group)) throw new Error(`${model.name} の KAM の群は学習器を持たない`);
  const { cell, form, train_comp } = model.config;
  return {
    format: FORMATS.part,
    device: trainDevice(model),
    kind: 'part',
    cell,
    form,
    group,
    data: hashOf(dataSpec(model)),
    train_comp,
  };
}

/** モデルの spec。組むだけなので CPU。学習器を組むモデルは学習器の並び（群の順）を、KAM のモデルは群と学習データを持つ。 */
export function modelSpec(model: ModelDecl): Spec {
  const { cell, form, absorbed, groups, train_comp } = model.config;
  const base = {
    format: FORMATS.model,
    device: 'cpu' as const,
    kind: 'model' as const,
    cell,
    form,
    ...(absorbed ? { absorbed } : {}),
  };
  if (kamModel(model)) return { ...base, parts: [], kam: { groups, data: hashOf(dataSpec(model)), train_comp } };
  return { ...base, parts: groups.map(group => hashOf(partSpec(model, group))) };
}

/** 学習データのファイルの名前。中身から決める（traub.train.duration-9000）。 */
export function dataName(model: ModelDecl): string {
  const { target, current_type, current_params } = model.config.data;
  return [target, current_type, ...Object.entries(current_params).map(([k, v]) => `${k}-${v}`)].join('.');
}

/** 学習器のファイルの名前。学習データと同じく中身から決める：圧縮器と右辺（同じなら 1 つ）、潜在の次元（1 でなければ）、群の状態を小文字で + につないだもの（pca-sindy.h+n、kam.h+n、ae-sindy-d5.m+h+n+a+b+c）。 */
export function partName(group: Group): string {
  const kind = group.compressor === group.rhs ? group.compressor : `${group.compressor}-${group.rhs}`;
  return `${kind}${group.d === 1 ? '' : `-d${group.d}`}.${group.gates.map(gate => gate.toLowerCase()).join('+')}`;
}

/** 系列の入力（電流と掃引）。置換範囲はモデルの側に入る。 */
export function inputOf(series: Series): Input {
  const { replace_targets: _targets, ...input } = series;
  return input;
}

/** 系列をモデル（ハッシュ）で置き換えて回す spec。surrogate が null なら元のモデル。1 細胞を順に進めるので CPU（GPU は 1 刻みの命令の往復で遅い）。 */
export function simSpec(series: Series, surrogate: string | null): Spec {
  const model = surrogate === null ? null : { surrogate, replace_targets: series.replace_targets };
  return { format: FORMATS.sim, device: 'cpu', kind: 'sim', input: inputOf(series), model };
}

/** GPU の実測の spec。細胞を並べて GPU で回す。surrogate はモデルのハッシュで、null なら元のモデル。 */
export function benchSpec(series: Series, surrogate: string | null, bench: Bench): Spec {
  const model = surrogate === null ? null : { surrogate, replace_targets: series.replace_targets };
  return {
    format: FORMATS.bench,
    device: 'gpu',
    kind: 'bench',
    input: inputOf(series),
    model,
    cells: bench.cells,
    ncu: bench.ncu,
  };
}

/** 実験が送る spec.json を、resolved/ の下のパス → 中身で並べる。 */
export function resolved(plan: GroupedPlan): Record<string, Spec> {
  const files: Record<string, Spec> = {};
  /** パスに spec を置く。同じパスに違う中身が来たら投げる（名前に出ない違いがあり、片方が黙って消えて指す側が引けなくなるのを防ぐ）。 */
  const put = (path: string, spec: Spec) => {
    if (path in files && hashOf(files[path]) !== hashOf(spec))
      throw new Error(`resolved/${path} に違う中身の spec が二つ来た`);
    files[path] = spec;
  };
  const modelHash = (model: string) => hashOf(modelSpec(modelOf(model)));
  for (const m of plan.models) {
    put(`${plan.id}/data/${dataName(m)}.json`, dataSpec(m));
    if (!kamModel(m))
      for (const group of m.config.groups) put(`${plan.id}/parts/${partName(group)}.json`, partSpec(m, group));
    put(`${plan.id}/models/${m.name}.json`, modelSpec(m));
  }
  for (const [name, s] of Object.entries(plan.series)) files[`${plan.id}/sim/${name}/original.json`] = simSpec(s, null);
  for (const p of plan.pairs)
    files[`${plan.id}/sim/${p.series}/${p.model}.json`] = simSpec(plan.series[p.series], modelHash(p.model));
  for (const b of plan.benches)
    files[`${plan.id}/bench/${b.name}.json`] = benchSpec(
      seriesOf(b.series),
      b.model === null ? null : modelHash(b.model),
      b,
    );
  return files;
}

/** テスト専用の学習の名前 → 設定。計算する機械の pytest が読む（宣言を送らずに済むよう、組んだ設定を値で渡す）。 */
function testConfigs(): Record<string, ModelDecl['config']> {
  return Object.fromEntries(Object.entries(TESTS).map(([name, train]) => [name, train.config]));
}

/** 宣言の写し。実験群と実験を宣言の順に、比べ方の物差しと一緒に。 */
export function plans(): Plans {
  return {
    groups: Object.entries(GROUPS).map(([name, description]) => ({ name, description })),
    experiments: EXPERIMENTS,
    measures: MEASURES,
  };
}

/** 全部の実験の、送る spec.json と、テスト専用の学習（tests.json）と、宣言の写し（plans.json）。 */
function expected(): Record<string, unknown> {
  const files: Record<string, unknown> = { 'tests.json': testConfigs(), 'plans.json': plans() };
  for (const e of EXPERIMENTS) Object.assign(files, resolved(e));
  return files;
}

/** resolved/ の下のファイルを全部（パスは resolved/ から）。 */
export function resolvedFiles(dir = RESOLVED): string[] {
  if (!existsSync(dir)) return [];
  return readdirSync(dir, { recursive: true, withFileTypes: true })
    .filter(f => f.isFile())
    .map(f => relative(dir, join(f.parentPath, f.name)))
    .sort();
}

/** resolved/ のうち、今の宣言と違うファイル（足りない・余分・中身が違う）。空なら新しい。 */
export function stale(dir = RESOLVED): string[] {
  const want = expected();
  const have = new Set(resolvedFiles(dir));
  const found: string[] = [];
  for (const path of have) if (!(path in want)) found.push(`余分 ${path}`);
  for (const [path, spec] of Object.entries(want)) {
    if (!have.has(path)) found.push(`足りない ${path}`);
    else if (JSON.stringify(JSON.parse(readFileSync(join(dir, path), 'utf8'))) !== JSON.stringify(spec))
      found.push(`古い ${path}`);
  }
  return found.sort();
}

/** resolved/ を今の宣言から書き直す。宣言に無いファイルは消す。 */
export function write(dir = RESOLVED): void {
  rmSync(dir, { recursive: true, force: true });
  for (const [path, spec] of Object.entries(expected())) {
    mkdirSync(dirname(join(dir, path)), { recursive: true });
    writeFileSync(join(dir, path), `${JSON.stringify(spec, null, 2)}\n`);
  }
}

function main([name]: string[]): void {
  if (name === 'write') write();
  else if (name === 'check') {
    const found = stale();
    if (found.length === 0) return;
    console.error(`resolved/ が宣言と違う（bun local/resolve.ts write で書き直す）:\n  ${found.join('\n  ')}`);
    process.exitCode = 1;
  } else
    console.log(
      `仕方のプリセット: ${METHOD_NAMES.join(' ')}\nモデル: ${Object.keys(MODELS).join(' ')}\nテスト: ${Object.keys(TESTS).join(' ')}\n系列: ${Object.keys(SERIES).join(' ')}`,
    );
}

if (import.meta.main) main(process.argv.slice(2));
