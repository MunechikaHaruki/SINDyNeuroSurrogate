// 段取り: conf.ts の宣言から spec.json を組み、ハッシュを鍵に runs/<ハッシュ>/ を Python の計算の入口（neurosurrogate/run.py）で埋める。
// 同じ spec は同じハッシュになるので、ディレクトリがあれば回さない。回したあとは runs/index.json（図が読む run の一覧）を書き直す。
// 入口は二つで、どちらもここの関数を呼ぶ。エージェントは CLI（just train・just eval・just runs）、本人は図のボタン（api.ts）。
//   bun research/50-59-research/53-evaluation/orchestrate.ts train <preset>...
//   … eval <系列> <preset かハッシュ>...
//   … index | spec <preset> | series | list
// --runs <dir> で置き場を変えられる（just test が使う）。
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  closeSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  openSync,
  readFileSync,
  readdirSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { FORMAT, PRESETS, SERIES, type IndexEntry, type Job, type Series, type Spec, type State, type Status, type SurrogateConfig } from './conf.ts';

const REPO = fileURLToPath(new URL('../../..', import.meta.url));
/** 既定の run の置き場 */
export const RUNS = join(REPO, 'runs');
/** run.py が「置換できない」で返す終了コード */
const NOT_REPLACEABLE = 3;
/** run.py が「同じ run を別のプロセスが回している」で返す終了コード */
const RUNNING = 4;

/** 鍵を整列した JSON。同じ中身なら同じ文字列になる。 */
export function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (typeof value === 'object' && value !== null) {
    const keys = Object.keys(value).sort();
    return `{${keys.map(k => `${JSON.stringify(k)}:${canonical((value as Record<string, unknown>)[k])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

function digest(value: unknown): string {
  return createHash('sha1').update(canonical(value)).digest('hex').slice(0, 12);
}

/** spec の同一性の鍵。ハッシュを組むのはここだけ。 */
export function hashOf(spec: Spec): string {
  return digest(spec);
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

/** runs/ の下の完成した run を新しい順に。書きかけ（.partial）と印のファイルは数えない。 */
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

export function writeIndex(runs: string): void {
  if (existsSync(runs)) writeFileSync(join(runs, 'index.json'), JSON.stringify(scan(runs)));
}

/** 置換できないと分かった組の印。置換できるかは Python だけが知るので、一度聞いた答えをハッシュで覚えておく。 */
function unreplaceableMark(runs: string, hash: string): string {
  return join(runs, `${hash}.unreplaceable`);
}

/** spec の run を用意してハッシュを返す。あれば回さない。置換できなければ undefined。 */
function ensure(runs: string, spec: Spec, name: string): string | undefined {
  const hash = hashOf(spec);
  const root = join(runs, hash);
  if (existsSync(unreplaceableMark(runs, hash))) return undefined;
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
  if (exitCode === NOT_REPLACEABLE) {
    writeFileSync(unreplaceableMark(runs, hash), '');
    return undefined;
  }
  if (exitCode === RUNNING) {
    // 別の段取りが同じ run を回している（原系を共有する二つの評価など）。終わるのを待って使う。
    console.log(`${hash} ${name}: 別のプロセスが回しているので待つ`);
    while (!existsSync(root) && running(join(runs, `${hash}.partial`))) Bun.sleepSync(5000);
    return ensure(runs, spec, name);
  }
  if (exitCode !== 0) throw new Error(`${hash} ${name}: run.py が ${exitCode} で落ちた`);
  return hash;
}

function presetOf(name: string) {
  const runs = PRESETS[name];
  if (!runs) throw new Error(`preset ${name} が無い（${Object.keys(PRESETS).join(', ')}）`);
  return runs;
}

function seriesOf(name: string): Series {
  const series = SERIES[name];
  if (!series) throw new Error(`系列 ${name} が無い（${Object.keys(SERIES).join(', ')}）`);
  return series;
}

/** 学習 run を指す引数（preset 名か、ハッシュかその先頭）→ 学習 run のハッシュと名前。preset は無ければ学習する。 */
function trainRuns(runs: string, selector: string): { hash: string; name: string }[] {
  if (selector in PRESETS)
    return presetOf(selector).map(t => ({ hash: ensure(runs, trainSpec(t.config), t.name)!, name: t.name }));
  const hits = scan(runs).filter(e => e.kind === 'train' && e.hash.startsWith(selector));
  if (hits.length !== 1) throw new Error(`学習 run ${selector} が ${hits.length} 本に当たる`);
  return [{ hash: hits[0].hash, name: hits[0].name }];
}

/** preset の学習 run を全部用意する。 */
export function train(runs: string, preset: string): void {
  for (const t of presetOf(preset)) ensure(runs, trainSpec(t.config), t.name);
  writeIndex(runs);
}

/** 系列を原系と、選んだ学習 run（preset かハッシュ）で回す。置換できない学習 run は回さない。 */
export function evaluate(runs: string, name: string, selectors: string[]): void {
  const series = seriesOf(name);
  ensure(runs, originalSpec(series), name);
  writeIndex(runs);
  for (const selector of selectors)
    for (const t of trainRuns(runs, selector))
      if (ensure(runs, replacedSpec(series, t.hash), name) === undefined)
        console.log(`${t.name} は ${name} を置換できないので回さない`);
  writeIndex(runs);
}

// ---- 図のボタンから回す ----


/** pid のプロセスが生きているか。 */
function alive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

/** 書きかけの run（<ハッシュ>.partial）を、書いているプロセスが生きているか。run.py が中に pid を置く。 */
function running(partial: string): boolean {
  const file = join(partial, 'pid');
  return existsSync(file) && alive(Number(readFileSync(file, 'utf8')));
}

/** 図から起こした段取りの置き場。仕事の鍵は {何を, どれを} のハッシュで、同じ組は同じ鍵になる。 */
function jobFile(runs: string, job: object, ext: string): string {
  return join(runs, '.jobs', `${digest(job)}.${ext}`);
}

function jobRunning(runs: string, job: object): boolean {
  const file = jobFile(runs, job, 'pid');
  return existsSync(file) && alive(Number(readFileSync(file, 'utf8')));
}

/** run 1 本の状態。 */
function runState(runs: string, hash: string): State {
  if (existsSync(unreplaceableMark(runs, hash))) return 'unreplaceable';
  if (existsSync(join(runs, hash))) return 'done';
  return running(join(runs, `${hash}.partial`)) ? 'running' : 'todo';
}

/** 宣言の preset と、preset × 系列の組の状態。 */
export function status(runs: string): Status {
  const trains: Status['trains'] = [];
  const pairs: Status['pairs'] = [];
  for (const [preset, list] of Object.entries(PRESETS)) {
    const hashes = list.map(t => hashOf(trainSpec(t.config)));
    const job = jobRunning(runs, { action: 'train', preset });
    const trainStates = hashes.map(h => runState(runs, h));
    if (job || trainStates.includes('running')) trains.push({ preset, state: 'running' });
    else trains.push({ preset, state: trainStates.every(s => s === 'done') ? 'done' : 'todo' });
    for (const [name, series] of Object.entries(SERIES)) {
      const replaced = hashes.map(h => runState(runs, hashOf(replacedSpec(series, h))));
      const parts = [runState(runs, hashOf(originalSpec(series))), ...trainStates, ...replaced];
      let state: State = 'todo';
      if (jobRunning(runs, { action: 'eval', series: name, preset }) || parts.includes('running')) state = 'running';
      else if (replaced.every(s => s === 'unreplaceable')) state = 'unreplaceable';
      else if (replaced.every(s => s === 'done' || s === 'unreplaceable')) state = 'done';
      pairs.push({ series: name, preset, state });
    }
  }
  return { trains, pairs };
}

/** 段取りを切り離した子プロセスで起こしてすぐ返す（開発サーバーが落ちても道連れにしない）。同じ組が回っていれば起こさない。 */
export function launch(runs: string, job: Job): boolean {
  presetOf(job.preset);
  if (job.action === 'eval') seriesOf(job.series);
  if (jobRunning(runs, job)) return false;
  mkdirSync(join(runs, '.jobs'), { recursive: true });
  const log = openSync(jobFile(runs, job, 'log'), 'w');
  const args = job.action === 'train' ? ['train', job.preset] : ['eval', job.series, job.preset];
  const child = spawn(process.execPath, [fileURLToPath(import.meta.url), ...args, '--runs', runs], {
    cwd: REPO,
    detached: true,
    stdio: ['ignore', log, log],
  });
  closeSync(log);
  writeFileSync(jobFile(runs, job, 'pid'), String(child.pid));
  child.unref();
  return true;
}

function main(argv: string[]): void {
  const at = argv.indexOf('--runs');
  const runs = at >= 0 ? resolve(argv.splice(at, 2)[1]) : RUNS;
  const [command, ...rest] = argv;
  if (command === 'spec') console.log(JSON.stringify(presetOf(rest[0]).map(t => t.config)));
  else if (command === 'series') console.log(JSON.stringify(SERIES));
  else if (command === 'list')
    for (const e of scan(runs)) console.log(`${e.hash} ${e.kind.padEnd(8)} ${e.created} ${e.name}`);
  else if (command === 'train') for (const preset of rest) train(runs, preset);
  else if (command === 'eval') evaluate(runs, rest[0], rest.slice(1));
  else if (command === 'index') writeIndex(runs);
  else console.log(`preset: ${Object.keys(PRESETS).join(' ')}\n系列: ${Object.keys(SERIES).join(' ')}`);
}

if (import.meta.main) main(process.argv.slice(2));
