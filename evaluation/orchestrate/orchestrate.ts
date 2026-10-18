// 段取り: conf.ts の宣言から spec.json を組み、ハッシュで決めた置き場を Python の計算の入口（neurosurrogate/run.py）で埋める。
// 置き場は学習が runs/models/<ハッシュ>/、入力 × モデルが runs/inputs/<入力のハッシュ>/<original かモデルの鍵>/。
// 同じ spec は同じ置き場になるので、ディレクトリがあれば回さない。回したあとは runs/index.json（図が読む run の一覧）を書き直す。
// 入口は二つで、どちらもここの関数を呼ぶ。本人は操作盤の冊のボタン（api.ts）、エージェントはこの CLI。人は CLI を使わない。
//   bun evaluation/orchestrate/orchestrate.ts train <preset>...
//   … eval <系列> <preset かハッシュ>...
//   … index | spec <preset> | series | list
// --runs <dir> で置き場を変えられる（just test が使う）。--job <鍵> は操作盤から起こしたときに launch が付け、終わりを .jobs/<鍵>.exit に書く。
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
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { EXPERIMENTS, PRESETS, SERIES } from './conf.ts';
import { FORMAT, type ExperimentStatus, type IndexEntry, type Input, type Job, type JobState, type RunStatus, type Series, type Spec, type State, type Status, type SurrogateConfig } from './spec.ts';

const REPO = fileURLToPath(new URL('../..', import.meta.url));
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

/** 系列の入力（電流と掃引）。置換範囲はモデルの側に入る。 */
export function inputOf(series: Series): Input {
  const { replace_targets: _targets, ...input } = series;
  return input;
}

/** 系列を学習 run（ハッシュ）で置き換えて回す spec。train が null なら元のモデル。 */
export function simSpec(series: Series, train: string | null): Spec {
  const model = train === null ? null : { train, replace_targets: series.replace_targets };
  return { format: FORMAT, kind: 'sim', input: inputOf(series), model };
}

/** spec の置き場（runs/ の下のパス）。置換範囲だけが違う系列は同じ入力のディレクトリに並ぶ。 */
export function dirOf(spec: Spec): string {
  if (spec.kind === 'train') return `models/${hashOf(spec)}`;
  const input = digest({ format: spec.format, input: spec.input });
  return `inputs/${input}/${spec.model === null ? 'original' : digest(spec.model)}`;
}

/** 置き場の下の、名前に . を含まないディレクトリ。 */
function children(dir: string): string[] {
  if (!existsSync(dir)) return [];
  const names: string[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) if (entry.isDirectory() && !entry.name.includes('.')) names.push(entry.name);
  return names;
}

/** runs/ の下の完成した run を新しい順に。書きかけ（.partial）と印のファイルは数えない。 */
export function scan(runs: string): IndexEntry[] {
  const dirs: string[] = [];
  for (const name of children(join(runs, 'models'))) dirs.push(join(runs, 'models', name));
  for (const input of children(join(runs, 'inputs'))) for (const name of children(join(runs, 'inputs', input))) dirs.push(join(runs, 'inputs', input, name));
  const list: IndexEntry[] = [];
  for (const dir of dirs) {
    if (!existsSync(join(dir, 'meta.json'))) continue;
    const meta = JSON.parse(readFileSync(join(dir, 'meta.json'), 'utf8'));
    list.push({ id: relative(runs, dir), ...meta, spec: JSON.parse(readFileSync(join(dir, 'spec.json'), 'utf8')) });
  }
  return list.sort((a, b) => b.created.localeCompare(a.created));
}

export function writeIndex(runs: string): void {
  if (existsSync(runs)) writeFileSync(join(runs, 'index.json'), JSON.stringify(scan(runs)));
}

/** 置換できないと分かった組の印。置換できるかは Python だけが知るので、一度聞いた答えを置き場の隣に覚えておく。 */
function unreplaceableMark(runs: string, id: string): string {
  return join(runs, `${id}.unreplaceable`);
}

/** spec の run を用意して置き場（runs/ の下のパス）を返す。あれば回さない。置換できなければ undefined。 */
function ensure(runs: string, spec: Spec, name: string): string | undefined {
  const id = dirOf(spec);
  const root = join(runs, id);
  if (existsSync(unreplaceableMark(runs, id))) return undefined;
  if (existsSync(root)) {
    console.log(`${id} ${name}: 回してある`);
    return id;
  }
  if (spec.kind === 'sim') {
    mkdirSync(dirname(root), { recursive: true });
    writeFileSync(join(dirname(root), 'input.json'), JSON.stringify(spec.input, null, 2));
  }
  const file = join(mkdtempSync(join(tmpdir(), 'spec-')), 'spec.json');
  writeFileSync(file, JSON.stringify(spec, null, 2));
  console.log(`${id} ${name}: 回す`);
  const { exitCode } = Bun.spawnSync(['uv', 'run', 'python', '-m', 'neurosurrogate.run', file, root, '--name', name], {
    cwd: REPO,
    stdio: ['inherit', 'inherit', 'inherit'],
  });
  if (exitCode === NOT_REPLACEABLE) {
    writeFileSync(unreplaceableMark(runs, id), '');
    return undefined;
  }
  if (exitCode === RUNNING) {
    // 別の段取りが同じ run を回している（元のモデルを共有する二つの評価など）。終わるのを待って使う。
    console.log(`${id} ${name}: 別のプロセスが回しているので待つ`);
    while (!existsSync(root) && running(join(runs, `${id}.partial`))) Bun.sleepSync(5000);
    return ensure(runs, spec, name);
  }
  if (exitCode !== 0) throw new Error(`${id} ${name}: run.py が ${exitCode} で落ちた`);
  return id;
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
    return presetOf(selector).map(t => {
      ensure(runs, trainSpec(t.config), t.name);
      return { hash: hashOf(trainSpec(t.config)), name: t.name };
    });
  const hits = scan(runs).filter(e => e.kind === 'train' && e.id.startsWith(`models/${selector}`));
  if (hits.length !== 1) throw new Error(`学習 run ${selector} が ${hits.length} 本に当たる`);
  return [{ hash: hits[0].id.slice('models/'.length), name: hits[0].name }];
}

/** preset の学習 run を全部用意する。 */
export function train(runs: string, preset: string): void {
  for (const t of presetOf(preset)) ensure(runs, trainSpec(t.config), t.name);
  writeIndex(runs);
}

/** 系列を元のモデルと、選んだ学習 run（preset かハッシュ）で置き換えたモデルで回す。置換できない学習 run は回さない。 */
export function evaluate(runs: string, name: string, selectors: string[]): void {
  const series = seriesOf(name);
  ensure(runs, simSpec(series, null), name);
  writeIndex(runs);
  for (const selector of selectors)
    for (const t of trainRuns(runs, selector))
      if (ensure(runs, simSpec(series, t.hash), `${name} × ${t.name}`) === undefined)
        console.log(`${t.name} は ${name} を置換できないので回さない`);
  writeIndex(runs);
}

// ---- 操作盤のボタンから回す ----

/** 操作盤に見せるログの末尾の行数 */
const LOG_TAIL = 30;

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

/** 操作盤から起こした段取りの置き場 runs/.jobs/。仕事の鍵は {何を, どれを} のハッシュで、同じ組は同じ鍵になる。 */
function jobDir(runs: string): string {
  return join(runs, '.jobs');
}

function jobFile(runs: string, job: object, ext: string): string {
  return join(jobDir(runs), `${digest(job)}.${ext}`);
}

function jobRunning(runs: string, job: object): boolean {
  const file = jobFile(runs, job, 'pid');
  return existsSync(file) && alive(Number(readFileSync(file, 'utf8')));
}

/** ファイルの末尾 n 行。無ければ空。 */
function tail(file: string, n: number): string[] {
  if (!existsSync(file)) return [];
  const lines = readFileSync(file, 'utf8').trimEnd().split('\n');
  return lines.slice(Math.max(0, lines.length - n));
}

/** 操作盤から起こした仕事を新しい順に。<鍵>.json が仕事、.exit が終了コード、それが無くて .pid が生きていれば計算中。 */
export function jobs(runs: string): JobState[] {
  const dir = jobDir(runs);
  if (!existsSync(dir)) return [];
  const list: JobState[] = [];
  for (const name of readdirSync(dir)) {
    if (!name.endsWith('.json')) continue;
    const key = name.slice(0, -'.json'.length);
    const { job, started } = JSON.parse(readFileSync(join(dir, name), 'utf8')) as { job: Job; started: string };
    const exit = join(dir, `${key}.exit`);
    // 終了コードを先に見る。書き終えた子が回収されるまで pid は生きて見える。
    let state: JobState['state'] = 'failed';
    if (existsSync(exit)) state = readFileSync(exit, 'utf8').trim() === '0' ? 'done' : 'failed';
    else if (jobRunning(runs, job)) state = 'running';
    list.push({ job, started, state, log: tail(join(dir, `${key}.log`), LOG_TAIL) });
  }
  return list.sort((a, b) => b.started.localeCompare(a.started));
}

/** run 1 本の状態。 */
function runState(runs: string, id: string): State {
  if (existsSync(unreplaceableMark(runs, id))) return 'unreplaceable';
  if (existsSync(join(runs, id))) return 'done';
  return running(join(runs, `${id}.partial`)) ? 'running' : 'todo';
}

/** 宣言した run 1 本の名前・置き場・状態。 */
function runStatus(runs: string, spec: Spec, name: string): RunStatus {
  const id = dirOf(spec);
  return { name, id, state: runState(runs, id) };
}

/** 組の状態。置換系がどれか回っていれば計算中、全部が置換できなければ置換できない、全部そろえば済み。 */
function pairState(runs: string, job: Job, parts: State[], replaced: State[]): State {
  if (jobRunning(runs, job) || parts.includes('running')) return 'running';
  if (replaced.every(s => s === 'unreplaceable')) return 'unreplaceable';
  if (replaced.every(s => s === 'done' || s === 'unreplaceable')) return 'done';
  return 'todo';
}

/** 実験 1 つの preset・系列・組の、ハッシュと状態。 */
function experimentStatus(runs: string, name: string): ExperimentStatus {
  const experiment = EXPERIMENTS[name];
  const result: ExperimentStatus = { name, presets: [], series: [], pairs: [] };
  for (const series of experiment.series) result.series.push({ series, original: runStatus(runs, simSpec(seriesOf(series), null), series) });
  for (const preset of experiment.presets) {
    const trains: RunStatus[] = [];
    const hashes: string[] = [];
    for (const t of presetOf(preset)) {
      trains.push(runStatus(runs, trainSpec(t.config), t.name));
      hashes.push(hashOf(trainSpec(t.config)));
    }
    const trainStates = trains.map(t => t.state);
    let state: State = trainStates.every(s => s === 'done') ? 'done' : 'todo';
    if (jobRunning(runs, { action: 'train', preset }) || trainStates.includes('running')) state = 'running';
    result.presets.push({ preset, state, runs: trains });
    for (const entry of result.series) {
      const replaced: RunStatus[] = [];
      trains.forEach((t, i) => replaced.push(runStatus(runs, simSpec(seriesOf(entry.series), hashes[i]), t.name)));
      const states = replaced.map(r => r.state);
      const job: Job = { action: 'eval', series: entry.series, preset };
      result.pairs.push({ series: entry.series, preset, state: pairState(runs, job, [entry.original.state, ...trainStates, ...states], states), runs: replaced });
    }
  }
  return result;
}

/** 宣言した実験ごとの状態と、操作盤から起こした仕事。 */
export function status(runs: string): Status {
  const experiments: ExperimentStatus[] = [];
  for (const name of Object.keys(EXPERIMENTS)) experiments.push(experimentStatus(runs, name));
  return { experiments, jobs: jobs(runs) };
}

/** 段取りを切り離した子プロセスで起こしてすぐ返す（開発サーバーが落ちても道連れにしない）。同じ組が回っていれば起こさない。 */
export function launch(runs: string, job: Job): boolean {
  presetOf(job.preset);
  if (job.action === 'eval') seriesOf(job.series);
  if (jobRunning(runs, job)) return false;
  mkdirSync(jobDir(runs), { recursive: true });
  rmSync(jobFile(runs, job, 'exit'), { force: true });
  writeFileSync(jobFile(runs, job, 'json'), JSON.stringify({ job, started: new Date().toISOString() }));
  const log = openSync(jobFile(runs, job, 'log'), 'w');
  const args = job.action === 'train' ? ['train', job.preset] : ['eval', job.series, job.preset];
  const child = spawn(process.execPath, [fileURLToPath(import.meta.url), ...args, '--runs', runs, '--job', digest(job)], {
    cwd: REPO,
    detached: true,
    stdio: ['ignore', log, log],
  });
  closeSync(log);
  writeFileSync(jobFile(runs, job, 'pid'), String(child.pid));
  child.unref();
  return true;
}

/** 引数から `--名前 値` を抜き出す。無ければ undefined。 */
function option(argv: string[], flag: string): string | undefined {
  const at = argv.indexOf(flag);
  return at >= 0 ? argv.splice(at, 2)[1] : undefined;
}

function main(argv: string[]): void {
  const dir = option(argv, '--runs');
  const runs = dir ? resolve(dir) : RUNS;
  const job = option(argv, '--job');
  if (!job) return command(runs, argv);
  // 操作盤から起こされた。落ちても終了コードを残し、操作盤が失敗として出せるようにする。
  let code = 0;
  try {
    command(runs, argv);
  } catch (e) {
    console.error(e instanceof Error ? e.message : e);
    code = 1;
  }
  writeFileSync(join(jobDir(runs), `${job}.exit`), String(code));
  process.exitCode = code;
}

function command(runs: string, argv: string[]): void {
  const [name, ...rest] = argv;
  if (name === 'spec') console.log(JSON.stringify(presetOf(rest[0]).map(t => t.config)));
  else if (name === 'series') console.log(JSON.stringify(SERIES));
  else if (name === 'list')
    for (const e of scan(runs)) console.log(`${e.id} ${e.kind.padEnd(8)} ${e.created} ${e.name}`);
  else if (name === 'train') for (const preset of rest) train(runs, preset);
  else if (name === 'eval') evaluate(runs, rest[0], rest.slice(1));
  else if (name === 'index') writeIndex(runs);
  else console.log(`preset: ${Object.keys(PRESETS).join(' ')}\n系列: ${Object.keys(SERIES).join(' ')}`);
}

if (import.meta.main) main(process.argv.slice(2));
