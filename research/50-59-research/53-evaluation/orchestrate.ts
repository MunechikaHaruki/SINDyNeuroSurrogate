// 段取り: conf.ts の宣言から spec.json を組み、ハッシュを鍵に runs/<ハッシュ>/ を Python の計算の入口（neurosurrogate/run.py）で埋める。
// 同じ spec は同じハッシュになるので、ディレクトリがあれば回さない。回したあとは runs/index.json（図が読む run の一覧）を書き直す。
// 入口は repo の justfile（just train・just eval・just runs）。中身は
//   bun research/50-59-research/53-evaluation/orchestrate.ts train <preset>...
//   … eval <系列> <preset かハッシュ>...
//   … index | spec <preset> | series | list
// --runs <dir> で置き場を変えられる（just test が使う）。
import { createHash } from 'node:crypto';
import { existsSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { FORMAT, PRESETS, SERIES, type IndexEntry, type Series, type Spec, type SurrogateConfig } from './conf.ts';

const REPO = resolve(import.meta.dirname, '../../..');
/** run.py が「置換できない」で返す終了コード */
const NOT_REPLACEABLE = 3;

/** 鍵を整列した JSON。同じ中身なら同じ文字列になる。 */
export function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (typeof value === 'object' && value !== null) {
    const keys = Object.keys(value).sort();
    return `{${keys.map(k => `${JSON.stringify(k)}:${canonical((value as Record<string, unknown>)[k])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

/** spec の同一性の鍵。ハッシュを組むのはここだけ。 */
export function hashOf(spec: Spec): string {
  return createHash('sha1').update(canonical(spec)).digest('hex').slice(0, 12);
}

export function trainSpec(config: SurrogateConfig): Spec {
  return { format: FORMAT, kind: 'train', surrogate: config };
}

/** 原系は置換範囲を含まない。置換範囲だけが違う対照系列どうしで原系 run を共有する。 */
export function originalSpec(series: Series): Spec {
  const { replace_targets: _targets, ...rest } = series;
  return { format: FORMAT, kind: 'original', series: rest };
}

export function replacedSpec(series: Series, train: string): Spec {
  return { format: FORMAT, kind: 'replaced', series, original: hashOf(originalSpec(series)), train };
}

/** runs/ の下の完成した run を新しい順に。書きかけ（.partial）は数えない。 */
export function scan(runs: string): IndexEntry[] {
  if (!existsSync(runs)) return [];
  const list: IndexEntry[] = [];
  for (const entry of readdirSync(runs, { withFileTypes: true })) {
    if (!entry.isDirectory() || entry.name.includes('.')) continue;
    const dir = join(runs, entry.name);
    if (!existsSync(join(dir, 'meta.json'))) continue;
    const meta = JSON.parse(readFileSync(join(dir, 'meta.json'), 'utf8'));
    list.push({ hash: entry.name, ...meta, spec: JSON.parse(readFileSync(join(dir, 'spec.json'), 'utf8')) });
  }
  return list.sort((a, b) => b.created.localeCompare(a.created));
}

function writeIndex(runs: string): void {
  if (existsSync(runs)) writeFileSync(join(runs, 'index.json'), JSON.stringify(scan(runs)));
}

/** spec の run を用意してハッシュを返す。あれば回さない。置換できなければ undefined。 */
function ensure(runs: string, spec: Spec, name: string): string | undefined {
  const hash = hashOf(spec);
  const root = join(runs, hash);
  if (existsSync(root)) {
    console.log(`${hash} ${name}: 回してある`);
    return hash;
  }
  const file = join(mkdtempSync(join(tmpdir(), 'spec-')), 'spec.json');
  writeFileSync(file, JSON.stringify(spec, null, 2));
  console.log(`${hash} ${name}: 回す`);
  const { exitCode } = Bun.spawnSync(['uv', 'run', 'python', '-m', 'neurosurrogate.run', file, root, '--name', name], {
    cwd: REPO,
    stdio: ['inherit', 'inherit', 'inherit'],
  });
  if (exitCode === NOT_REPLACEABLE) return undefined;
  if (exitCode !== 0) throw new Error(`${hash} ${name}: run.py が ${exitCode} で落ちた`);
  return hash;
}

function presetOf(name: string) {
  const runs = PRESETS[name];
  if (!runs) throw new Error(`preset ${name} が無い（${Object.keys(PRESETS).join(', ')}）`);
  return runs;
}

/** 学習 run を指す引数（preset 名か、ハッシュかその先頭）→ 学習 run のハッシュと名前。preset は無ければ学習する。 */
function trainRuns(runs: string, selector: string): { hash: string; name: string }[] {
  if (selector in PRESETS)
    return presetOf(selector).map(t => ({ hash: ensure(runs, trainSpec(t.config), t.name)!, name: t.name }));
  const hits = scan(runs).filter(e => e.kind === 'train' && e.hash.startsWith(selector));
  if (hits.length !== 1) throw new Error(`学習 run ${selector} が ${hits.length} 本に当たる`);
  return [{ hash: hits[0].hash, name: hits[0].name }];
}

function evaluate(runs: string, name: string, selectors: string[]): void {
  const series = SERIES[name];
  if (!series) throw new Error(`系列 ${name} が無い（${Object.keys(SERIES).join(', ')}）`);
  ensure(runs, originalSpec(series), name);
  for (const selector of selectors)
    for (const train of trainRuns(runs, selector))
      if (ensure(runs, replacedSpec(series, train.hash), name) === undefined)
        console.log(`${train.name} は ${name} を置換できないので回さない`);
}

function main(argv: string[]): void {
  const at = argv.indexOf('--runs');
  const runs = at >= 0 ? resolve(argv.splice(at, 2)[1]) : join(REPO, 'runs');
  const [command, ...rest] = argv;
  if (command === 'spec') return console.log(JSON.stringify(presetOf(rest[0]).map(t => t.config)));
  if (command === 'series') return console.log(JSON.stringify(SERIES));
  if (command === 'list') {
    for (const e of scan(runs)) console.log(`${e.hash} ${e.kind.padEnd(8)} ${e.created} ${e.name}`);
    return;
  }
  if (command === 'train')
    for (const preset of rest) for (const t of presetOf(preset)) ensure(runs, trainSpec(t.config), t.name);
  else if (command === 'eval') evaluate(runs, rest[0], rest.slice(1));
  else if (command !== 'index')
    return console.log(`preset: ${Object.keys(PRESETS).join(' ')}\n系列: ${Object.keys(SERIES).join(' ')}`);
  writeIndex(runs);
}

if (import.meta.main) main(process.argv.slice(2));
