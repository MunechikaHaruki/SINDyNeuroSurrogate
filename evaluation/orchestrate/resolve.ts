// 宣言（config/ と ../experiments/）から、実験ごとに送る spec.json を組んで resolved/ に置く。実行（orchestrate.ts）は宣言を読まず、ここが書いた resolved/ だけを回す。
// パスは識別のための名前で、送るのは中身だけ：`<実験>/train/<学習 run>.json`、`<実験>/sim/<系列>/original.json`（元のモデル）と `<学習 run>.json`、`<実験>/bench/<GPU の実測>.json`。
//   bun evaluation/orchestrate/resolve.ts write        resolved/ を今の宣言から書き直す
//   … check                                          resolved/ が今の宣言と同じか（違えば挙げて終了コード 1。remote.sh up が送る前に見る）
//   … spec <学習 run> | series                       宣言した学習 run の設定と系列（tests/ が読む）。引数なしで名前の一覧
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { METHOD_NAMES, TESTS } from '../experiments/presets.ts';
import { EXPERIMENTS, SERIES, TRAINS, seriesOf, trainOf, type ThemedPlan } from '../experiments/experiments.ts';
import { RESOLVED, benchSpec, hashOf, simSpec, trainSpec } from './orchestrate.ts';
import type { Spec } from './spec.ts';

/** 実験が送る spec.json を、resolved/ の下のパス → 中身で並べる。 */
export function resolved(plan: ThemedPlan): Record<string, Spec> {
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

/** 全部の実験の、送る spec.json。 */
function expected(): Record<string, Spec> {
  const files: Record<string, Spec> = {};
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

function main([name, ...rest]: string[]): void {
  if (name === 'write') write();
  else if (name === 'check') {
    const found = stale();
    if (found.length === 0) return;
    console.error(
      `resolved/ が宣言と違う（bun evaluation/orchestrate/resolve.ts write で書き直す）:\n  ${found.join('\n  ')}`,
    );
    process.exitCode = 1;
  } else if (name === 'spec') console.log(JSON.stringify(trainOf(rest[0]).config));
  else if (name === 'series') console.log(JSON.stringify(SERIES));
  else
    console.log(
      `仕方のプリセット: ${METHOD_NAMES.join(' ')}\n学習 run（実験とテスト）: ${Object.keys(TRAINS).join(' ')}\nテスト: ${Object.keys(TESTS).join(' ')}\n系列: ${Object.keys(SERIES).join(' ')}`,
    );
}

if (import.meta.main) main(process.argv.slice(2));
