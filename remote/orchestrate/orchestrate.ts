// 段取り: resolved/ の spec.json（宣言から resolve.ts が組んだ、送るものそのもの）を、ハッシュで決めた置き場で Python の計算の入口（neurosurrogate/run.py）に回す。宣言は読まない。
// 置き場は学習が runs/models/<ハッシュ>/、入力 × モデルが runs/inputs/<入力のハッシュ>/<original かモデルの鍵>/、GPU の実測が runs/bench/<ハッシュ>/。
// 同じ spec は同じ置き場になるので、ディレクトリがあれば回さない。回したあとは runs/index.json（図が読む run の一覧）を書き直す。
// 入口は二つで、どちらもここの関数を呼ぶ。外の画面は export した関数を、手元ではこの CLI を使う。
//   bun remote/orchestrate/orchestrate.ts run <resolved/ の下のパス>...   その spec を回す。評価と GPU の実測は、指す学習 run と元のモデルを先に回す
//   … index | list
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
import type { ExperimentStatus, IndexEntry, Job, JobState, RunStatus, Spec, State, Status } from '../../shared/spec.ts';
import { digest, hashOf } from '../../shared/hash.ts';

const REPO = fileURLToPath(new URL('../..', import.meta.url));
/** 計算する機械の側（Python の project と、このファイル）。Python はここで起こす。 */
const HERE = fileURLToPath(new URL('..', import.meta.url));
/** 既定の run の置き場。計算する機械のイメージは SINDY_RUNS で載せた場所を指す。 */
export const RUNS = process.env.SINDY_RUNS ?? join(REPO, 'runs');
/** run.py が「置換できない」で返す終了コード */
const NOT_REPLACEABLE = 3;
/** run.py が「同じ run を別のプロセスが回している」で返す終了コード */
const RUNNING = 4;

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
    cwd: HERE,
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

/** resolved/ の置き場。送る spec.json をそのまま置く（resolve.ts が書く）。 */
export const RESOLVED = join(REPO, 'resolved');

/** resolved/ の下のパスの spec.json。resolved/ の外を指すパスは弾く。 */
export function specAt(path: string): Spec {
  const file = resolve(RESOLVED, path);
  if (!file.startsWith(`${RESOLVED}/`) || !existsSync(file)) throw new Error(`resolved/${path} が無い`);
  return JSON.parse(readFileSync(file, 'utf8')) as Spec;
}

/** 学習 run のハッシュ → resolved/ の学習の spec.json（評価と GPU の実測が先に回す学習 run を引く）。 */
function trainAt(hash: string): { path: string; spec: Spec } {
  for (const entry of readdirSync(RESOLVED, { recursive: true, withFileTypes: true })) {
    if (!entry.isFile() || !entry.parentPath.endsWith('/train')) continue;
    const path = relative(RESOLVED, join(entry.parentPath, entry.name));
    const spec = specAt(path);
    if (hashOf(spec) === hash) return { path, spec };
  }
  throw new Error(`学習 run ${hash} の spec が resolved/ に無い`);
}

/** resolved/ の spec を 1 つ回す。評価と GPU の実測は、指す学習 run と（評価なら）同じ入力の元のモデルを先に回す。名前はパスから .json を除いたもの。 */
export function runSpec(runs: string, path: string): void {
  const spec = specAt(path);
  const name = path.replace(/\.json$/, '');
  if (spec.kind !== 'train' && spec.model !== null) {
    const train = trainAt(spec.model.train);
    ensure(runs, train.spec, train.path.replace(/\.json$/, ''));
  }
  if (spec.kind === 'sim' && spec.model !== null) ensure(runs, { ...spec, model: null }, `${name} の元のモデル`);
  if (ensure(runs, spec, name) === undefined) console.log(`${name}: 置換できないので回さない`);
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

/** resolved/ の spec 1 つの名前・置き場・状態。 */
function runStatus(runs: string, path: string, name: string): RunStatus {
  const id = dirOf(specAt(path));
  return { name, id, spec: path, state: runState(runs, id) };
}

/** ディレクトリの下の .json の名前（.json を除いたもの）を名前の順に。無ければ空。 */
function specNames(dir: string): string[] {
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter(f => f.endsWith('.json'))
    .map(f => f.slice(0, -'.json'.length))
    .sort();
}

/** ディレクトリの下のディレクトリの名前を名前の順に。無ければ空。 */
function subdirs(dir: string): string[] {
  if (!existsSync(dir)) return [];
  return readdirSync(dir, { withFileTypes: true })
    .filter(e => e.isDirectory())
    .map(e => e.name)
    .sort();
}

/** 仕事が回っていれば計算中、でなければ run の状態。 */
function withJob(runs: string, run: RunStatus): State {
  return jobRunning(runs, { spec: run.spec }) ? 'running' : run.state;
}

/** resolved/<実験>/ の spec の、置き場と状態。学習 run は train/、系列は sim/<系列>/ の original と学習 run の名前、GPU の実測は bench/。並びは名前の順で、宣言の順は使う側が持つ。 */
function experimentStatus(runs: string, name: string): ExperimentStatus {
  const root = join(RESOLVED, name);
  const result: ExperimentStatus = { name, models: [], series: [], pairs: [], benches: [] };
  for (const model of specNames(join(root, 'train'))) {
    const run = runStatus(runs, `${name}/train/${model}.json`, model);
    result.models.push({ model, state: withJob(runs, run), run });
  }
  for (const series of subdirs(join(root, 'sim'))) {
    const original = runStatus(runs, `${name}/sim/${series}/original.json`, series);
    result.series.push({ series, original });
    for (const model of specNames(join(root, 'sim', series))) {
      if (model === 'original') continue;
      const run = runStatus(runs, `${name}/sim/${series}/${model}.json`, model);
      const train = result.models.find(m => m.model === model)?.run.state;
      const running = jobRunning(runs, { spec: run.spec }) || [original.state, train, run.state].includes('running');
      result.pairs.push({ series, model, state: running ? 'running' : run.state, run });
    }
  }
  for (const bench of specNames(join(root, 'bench'))) {
    const run = runStatus(runs, `${name}/bench/${bench}.json`, `${name} ${bench}`);
    result.benches.push({ bench, state: withJob(runs, run), run });
  }
  return result;
}

/** resolved/ の実験ごとの状態と、外の画面から起こした仕事。実験は resolved/ の下のディレクトリで、宣言は読まない。 */
export function status(runs: string): Status {
  return { experiments: subdirs(RESOLVED).map(name => experimentStatus(runs, name)), jobs: jobs(runs) };
}

/** 段取りを切り離した子プロセスで起こしてすぐ返す（開発サーバーが落ちても道連れにしない）。同じ組が回っていれば起こさない。 */
export function launch(runs: string, job: Job): boolean {
  specAt(job.spec);
  if (jobRunning(runs, job)) return false;
  mkdirSync(jobDir(runs), { recursive: true });
  rmSync(jobFile(runs, job, 'exit'), { force: true });
  writeFileSync(jobFile(runs, job, 'json'), JSON.stringify({ job, started: new Date().toISOString() }));
  const log = openSync(jobFile(runs, job, 'log'), 'w');
  const args = ['run', job.spec];
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
  if (name === 'run') for (const path of rest) runSpec(runs, path);
  else if (name === 'list')
    for (const e of scan(runs)) console.log(`${e.id} ${e.kind.padEnd(8)} ${e.created} ${e.name}`);
  else if (name === 'index') writeIndex(runs);
  else console.log('run <resolved/ の下のパス>... | index | list');
}

if (import.meta.main) main(process.argv.slice(2));
