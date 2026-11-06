// 段取り: 実験（experiments/experiments.ts の解決済みの学習 run と系列）から spec.json を組み、ハッシュで決めた置き場を Python の計算の入口（neurosurrogate/run.py）で埋める。
// 置き場は学習が runs/models/<ハッシュ>/、入力 × モデルが runs/inputs/<入力のハッシュ>/<original かモデルの鍵>/。
// 同じ spec は同じ置き場になるので、ディレクトリがあれば回さない。回したあとは runs/index.json（図が読む run の一覧）を書き直す。
// 入口は二つで、どちらもここの関数を呼ぶ。外の画面は export した関数を、手元ではこの CLI を使う。
//   bun evaluation/orchestrate/orchestrate.ts train <学習 run>...        学習 run は実験の中の名前（kam_hn）かテストの名前
//   … eval <系列> <学習 run かハッシュ>...
//   … bench <実験> <GPU の実測の名前>...   GPU で細胞を並べて回し、時間・VRAM・L1 を測る
//   … index | spec <学習 run> | series | list
// --runs <dir> で置き場を変えられる。--job <鍵> は外の画面から起こしたときに launch が付け、終わりを .jobs/<鍵>.exit に書く。
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
import { METHOD_NAMES, TESTS } from './conf.ts';
import { EXPERIMENT_RUNS, SERIES, TRAINS, seriesOf, trainOf } from '../../experiments/experiments.ts';
import {
  FORMAT,
  type Bench,
  type Device,
  type Experiment,
  type ExperimentStatus,
  type IndexEntry,
  type Input,
  type Job,
  type JobState,
  type RunStatus,
  type Series,
  type Spec,
  type State,
  type Status,
  type Train,
} from './spec.ts';

const REPO = fileURLToPath(new URL('../..', import.meta.url));
/** 既定の run の置き場。計算する機械のイメージは SINDY_RUNS で載せた場所を指す。 */
export const RUNS = process.env.SINDY_RUNS ?? join(REPO, 'runs');
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

/** spec の置き場（runs/ の下のパス）。置換範囲だけが違う系列は同じ入力のディレクトリに並ぶ。 */
export function dirOf(spec: Spec): string {
  if (spec.kind === 'train') return `models/${hashOf(spec)}`;
  if (spec.kind === 'bench') return `bench/${hashOf(spec)}`;
  const input = digest({ format: spec.format, device: spec.device, input: spec.input });
  return `inputs/${input}/${spec.model === null ? 'original' : digest(spec.model)}`;
}

/** 置き場の下の、名前に . を含まないディレクトリ。 */
function children(dir: string): string[] {
  if (!existsSync(dir)) return [];
  const names: string[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true }))
    if (entry.isDirectory() && !entry.name.includes('.')) names.push(entry.name);
  return names;
}

/** runs/ の下の完成した run を新しい順に。書きかけ（.partial）と印のファイルは数えない。 */
export function scan(runs: string): IndexEntry[] {
  const dirs: string[] = [];
  for (const top of ['models', 'bench']) for (const name of children(join(runs, top))) dirs.push(join(runs, top, name));
  for (const input of children(join(runs, 'inputs')))
    for (const name of children(join(runs, 'inputs', input))) dirs.push(join(runs, 'inputs', input, name));
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

/** Python の計算の入口を起こすコマンド。計算する機械のイメージは venv の python を SINDY_PYTHON で渡す。 */
const PYTHON = process.env.SINDY_PYTHON?.split(' ') ?? ['uv', 'run', 'python'];

/** 回すコードの版。計算する機械へ送ったコード（.git が無い）は remote.sh が .code-version に書き、手元では git から組む。書きかけなら差分（未追跡のファイルは名前と中身）のハッシュを足す。 */
function codeVersion(): string {
  const sent = join(REPO, '.code-version');
  if (!existsSync(join(REPO, '.git')) && existsSync(sent)) return readFileSync(sent, 'utf8').trim();
  const git = (...args: string[]) => Bun.spawnSync(['git', ...args], { cwd: REPO }).stdout.toString();
  const head = git('rev-parse', '--short=12', 'HEAD').trim();
  let diff = git('diff', 'HEAD');
  for (const file of git('ls-files', '-z', '--others', '--exclude-standard').split('\0').filter(Boolean))
    diff += `${file}\n${readFileSync(join(REPO, file), 'utf8')}`;
  return diff === '' ? head : `${head}+${createHash('sha1').update(diff).digest('hex').slice(0, 12)}`;
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
  // GPU の実測は子プロセスが GPU を使うので、呼び出し元は VRAM を先取りしない
  const preallocate = spec.kind === 'bench' ? { XLA_PYTHON_CLIENT_PREALLOCATE: 'false' } : {};
  const env = {
    ...process.env,
    JAX_PLATFORMS: spec.device === 'gpu' ? 'cuda,cpu' : 'cpu',
    SINDY_CODE: codeVersion(),
    ...preallocate,
  };
  const { exitCode } = Bun.spawnSync([...PYTHON, '-m', 'neurosurrogate.run', file, root, '--name', name], {
    cwd: REPO,
    env,
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

/** 学習 run を指す引数（学習 run の名前か、ハッシュかその先頭）→ 学習 run のハッシュと名前。名前なら、無ければ学習する。 */
function trainRun(runs: string, selector: string): { hash: string; name: string } {
  if (!/^[0-9a-f]+$/.test(selector)) {
    const t = trainOf(selector);
    ensure(runs, trainSpec(t), t.name);
    return { hash: hashOf(trainSpec(t)), name: t.name };
  }
  const hits = scan(runs).filter(e => e.kind === 'train' && e.id.startsWith(`models/${selector}`));
  if (hits.length !== 1) throw new Error(`学習 run ${selector} が ${hits.length} 本に当たる`);
  return { hash: hits[0].id.slice('models/'.length), name: hits[0].name };
}

/** 学習 run を用意する。 */
export function train(runs: string, model: string): void {
  const t = trainOf(model);
  ensure(runs, trainSpec(t), t.name);
  writeIndex(runs);
}

/** 実験の GPU の実測を名前で引く。知らなければ投げる。 */
function benchOf(experiment: string, name: string): Bench {
  const found = EXPERIMENT_RUNS[experiment]?.benches?.find(b => b.name === name);
  if (!found) throw new Error(`実験 ${experiment} に GPU の実測 ${name} が無い`);
  return found;
}

/** GPU の実測の spec。置き換えたモデルなら学習 run の置き場のハッシュを指す。 */
function benchSpecOf(b: Bench): Spec {
  const train = b.model === null ? null : hashOf(trainSpec(trainOf(b.model)));
  return benchSpec(seriesOf(b.series), train, b);
}

/** 実験の GPU の実測を回す。置き換えたモデルなら、学習 run が無ければ先に学習する。 */
export function benchmark(runs: string, experiment: string, name: string): void {
  const b = benchOf(experiment, name);
  if (b.model !== null) train(runs, b.model);
  ensure(runs, benchSpecOf(b), `${experiment} ${name}`);
  writeIndex(runs);
}

/** 系列を元のモデルと、選んだ学習 run（名前かハッシュ）で置き換えたモデルで回す。置換できない学習 run は回さない。 */
export function evaluate(runs: string, name: string, selectors: string[]): void {
  const series = seriesOf(name);
  ensure(runs, simSpec(series, null), name);
  writeIndex(runs);
  for (const selector of selectors) {
    const t = trainRun(runs, selector);
    if (ensure(runs, simSpec(series, t.hash), `${name} × ${t.name}`) === undefined)
      console.log(`${t.name} は ${name} を置換できないので回さない`);
  }
  writeIndex(runs);
}

// ---- 外の画面から回す ----

/** 外の画面に返すログの末尾の行数 */
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

/** 外の画面から起こした段取りの置き場 runs/.jobs/。仕事の鍵は {何を, どれを} のハッシュで、同じ組は同じ鍵になる。 */
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

/** 外の画面から起こした仕事を新しい順に。<鍵>.json が仕事、.exit が終了コード、それが無くて .pid が生きていれば計算中。 */
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

/** 実験 1 つの学習 run・系列・組の、置き場と状態。起こした仕事が回っていれば計算中にする。 */
function experimentStatus(runs: string, name: string, experiment: Experiment): ExperimentStatus {
  const result: ExperimentStatus = { name, models: [], series: [], pairs: [], benches: [] };
  for (const { model, series } of experiment.pairs) {
    let found = result.models.find(m => m.model === model);
    if (!found) {
      const run = runStatus(runs, trainSpec(trainOf(model)), model);
      const state = jobRunning(runs, { action: 'train', model }) ? 'running' : run.state;
      found = { model, state, run };
      result.models.push(found);
    }
    let original = result.series.find(s => s.series === series)?.original;
    if (!original) {
      original = runStatus(runs, simSpec(seriesOf(series), null), series);
      result.series.push({ series, original });
    }
    const run = runStatus(runs, simSpec(seriesOf(series), hashOf(trainSpec(trainOf(model)))), model);
    const parts = [original.state, found.run.state, run.state];
    const state =
      jobRunning(runs, { action: 'eval', series, model }) || parts.includes('running') ? 'running' : run.state;
    result.pairs.push({ series, model, state, run });
  }
  for (const b of experiment.benches ?? []) {
    const run = runStatus(runs, benchSpecOf(b), `${name} ${b.name}`);
    const state = jobRunning(runs, { action: 'bench', experiment: name, bench: b.name }) ? 'running' : run.state;
    result.benches.push({ bench: b.name, state, run });
  }
  return result;
}

/** 渡された実験（回す学習 run と系列の組。持つのは使う側）ごとの状態と、外の画面から起こした仕事。 */
export function status(runs: string, experiments: Record<string, Experiment>): Status {
  const result: ExperimentStatus[] = [];
  for (const name of Object.keys(experiments)) result.push(experimentStatus(runs, name, experiments[name]));
  return { experiments: result, jobs: jobs(runs) };
}

/** 段取りを切り離した子プロセスで起こしてすぐ返す（開発サーバーが落ちても道連れにしない）。同じ組が回っていれば起こさない。 */
export function launch(runs: string, job: Job): boolean {
  if (job.action === 'bench') benchOf(job.experiment, job.bench);
  else trainOf(job.model);
  if (job.action === 'eval') seriesOf(job.series);
  if (jobRunning(runs, job)) return false;
  mkdirSync(jobDir(runs), { recursive: true });
  rmSync(jobFile(runs, job, 'exit'), { force: true });
  writeFileSync(jobFile(runs, job, 'json'), JSON.stringify({ job, started: new Date().toISOString() }));
  const log = openSync(jobFile(runs, job, 'log'), 'w');
  const args =
    job.action === 'train'
      ? ['train', job.model]
      : job.action === 'eval'
        ? ['eval', job.series, job.model]
        : ['bench', job.experiment, job.bench];
  const child = spawn(
    process.execPath,
    [fileURLToPath(import.meta.url), ...args, '--runs', runs, '--job', digest(job)],
    {
      cwd: REPO,
      detached: true,
      stdio: ['ignore', log, log],
    },
  );
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
  // 外の画面から起こされた。落ちても終了コードを残し、失敗として返せるようにする。
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
  if (name === 'spec') console.log(JSON.stringify(trainOf(rest[0]).config));
  else if (name === 'series') console.log(JSON.stringify(SERIES));
  else if (name === 'list')
    for (const e of scan(runs)) console.log(`${e.id} ${e.kind.padEnd(8)} ${e.created} ${e.name}`);
  else if (name === 'train') for (const model of rest) train(runs, model);
  else if (name === 'eval') evaluate(runs, rest[0], rest.slice(1));
  else if (name === 'bench') {
    if (rest.length < 2) throw new Error('bench <実験> <GPU の実測の名前>... と書く');
    for (const b of rest.slice(1)) benchmark(runs, rest[0], b);
  } else if (name === 'index') writeIndex(runs);
  else
    console.log(
      `仕方のプリセット: ${METHOD_NAMES.join(' ')}\n学習 run（実験とテスト）: ${Object.keys(TRAINS).join(' ')}\nテスト: ${Object.keys(TESTS).join(' ')}\n系列: ${Object.keys(SERIES).join(' ')}`,
    );
}

if (import.meta.main) main(process.argv.slice(2));
