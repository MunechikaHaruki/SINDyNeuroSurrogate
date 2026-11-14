// 手元の側の出口。宣言（../config/ と隣の TS）から、実験ごとに送る spec.json を組んで resolved/ に置く。テスト専用の学習の設定も resolved/tests.json（計算する機械の pytest が読む）に置く。計算する機械（../../remote/）は宣言を読まず、ここが書いた resolved/ だけを受け取って回す。
// パスは識別のための名前で、送るのは中身だけ：`<実験>/train/<学習 run>.json`、`<実験>/sim/<系列>/original.json`（元のモデル）と `<学習 run>.json`、`<実験>/bench/<GPU の実測>.json`。
//   bun local/resolve/resolve.ts write        resolved/ を今の宣言から書き直す
//   … check                                          resolved/ が今の宣言と同じか（違えば挙げて終了コード 1。remote.sh up が送る前に見る）
//   引数なしで名前の一覧
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { METHOD_NAMES, TESTS, type Series, type Train } from './presets.ts';
import { EXPERIMENTS, SERIES, TRAINS, seriesOf, trainOf, type GroupedPlan } from './experiments.ts';
import { hashOf } from '../../shared/hash.ts';
import { FORMAT, type Device, type Input, type Spec } from '../../shared/spec.ts';
import type { Bench } from './schema.ts';

/** 送る spec.json の置き場。 */
export const RESOLVED = fileURLToPath(new URL('../../resolved', import.meta.url));

/** 学習は GPU（AE の学習が速い）、テストの学習は GPU の無い機械でも回るよう CPU。 */
function trainDevice(train: Train): Device {
  return train.name.startsWith('_test_') ? 'cpu' : 'gpu';
}

export function trainSpec(train: Train): Spec {
  return { format: FORMAT, device: trainDevice(train), kind: 'train', surrogate: train.config };
}

/** 系列の入力（電流と掃引）。置換範囲はモデルの側に入る。 */
export function inputOf(series: Series): Input {
  const { replace_targets: _targets, ...input } = series;
  return input;
}

/** 系列を学習 run（ハッシュ）で置き換えて回す spec。train が null なら元のモデル。1 細胞を順に進めるので CPU（GPU は 1 刻みの命令の往復で遅い）。 */
export function simSpec(series: Series, train: string | null): Spec {
  const model = train === null ? null : { train, replace_targets: series.replace_targets };
  return { format: FORMAT, device: 'cpu', kind: 'sim', input: inputOf(series), model };
}

/** GPU の実測の spec。細胞を並べて GPU で回す。train は学習 run のハッシュで、null なら元のモデル。 */
export function benchSpec(series: Series, train: string | null, bench: Bench): Spec {
  const model = train === null ? null : { train, replace_targets: series.replace_targets };
  return {
    format: FORMAT,
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
  const trainHash = (model: string) => hashOf(trainSpec(trainOf(model)));
  for (const m of plan.models) files[`${plan.id}/train/${m.name}.json`] = trainSpec(m.train);
  for (const [name, s] of Object.entries(plan.series)) files[`${plan.id}/sim/${name}/original.json`] = simSpec(s, null);
  for (const p of plan.pairs)
    files[`${plan.id}/sim/${p.series}/${p.model}.json`] = simSpec(plan.series[p.series], trainHash(p.model));
  for (const b of plan.benches)
    files[`${plan.id}/bench/${b.name}.json`] = benchSpec(
      seriesOf(b.series),
      b.model === null ? null : trainHash(b.model),
      b,
    );
  return files;
}

/** テスト専用の学習の名前 → 設定。計算する機械の pytest が読む（宣言を送らずに済むよう、組んだ設定を値で渡す）。 */
function testConfigs(): Record<string, Train['config']> {
  return Object.fromEntries(Object.entries(TESTS).map(([name, train]) => [name, train.config]));
}

/** 全部の実験の、送る spec.json と、テスト専用の学習（tests.json）。 */
function expected(): Record<string, unknown> {
  const files: Record<string, unknown> = { 'tests.json': testConfigs() };
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
    console.error(
      `resolved/ が宣言と違う（bun local/resolve/resolve.ts write で書き直す）:\n  ${found.join('\n  ')}`,
    );
    process.exitCode = 1;
  } else
    console.log(
      `仕方のプリセット: ${METHOD_NAMES.join(' ')}\n学習 run（実験とテスト）: ${Object.keys(TRAINS).join(' ')}\nテスト: ${Object.keys(TESTS).join(' ')}\n系列: ${Object.keys(SERIES).join(' ')}`,
    );
}

if (import.meta.main) main(process.argv.slice(2));
