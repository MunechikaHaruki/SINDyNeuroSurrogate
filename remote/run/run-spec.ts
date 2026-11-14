// spec を回す。計算する機械の上の段取りの入口で、remote.sh run と、画面から起こした仕事（runs/.jobs/ に仕事の中身・pid・ログ・終了コードを置き、切り離した子プロセスでこのファイルを起こす。--job で終わりを .jobs/<鍵>.exit に書く）がこれを使う。
//   bun remote/run/run-spec.ts run <resolved/ の下のパス>...   その spec を回す
//   … list                                                      完成した run の一覧（--runs <dir> で置き場を変えられる）
// resolved/ の spec を読み、評価と GPU の実測なら指す学習 run と（評価なら）同じ入力の元のモデルを先に用意してから、
// Python の計算の入口（neurosurrogate/run.py）を起こして置き場を埋める。置き場があれば回さず、終了コードで置換できない組と、別のプロセスが回している run を見分ける。
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { closeSync, existsSync, mkdirSync, mkdtempSync, openSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Job, JobState, Spec } from '../../shared/spec.ts';
import { digest, hashOf } from '../../shared/hash.ts';
import { alive, dirOf, RESOLVED, RUNS, running, scan, specAt, unreplaceableMark } from './places.ts';

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

const REPO = fileURLToPath(new URL('../..', import.meta.url));
/** 計算する機械の側（Python の project）。Python はここで起こす。 */
const HERE = fileURLToPath(new URL('..', import.meta.url));
/** run.py が「置換できない」で返す終了コード */
const NOT_REPLACEABLE = 3;
/** run.py が「同じ run を別のプロセスが回している」で返す終了コード */
const RUNNING = 4;

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

// ---- 画面から起こした仕事


/** 外の画面に返すログの末尾の行数 */
const LOG_TAIL = 30;

/** 外の画面から起こした段取りの置き場 runs/.jobs/。仕事の鍵は {何を, どれを} のハッシュで、同じ組は同じ鍵になる。 */
function jobDir(runs: string): string {
  return join(runs, '.jobs');
}

function jobFile(runs: string, job: object, ext: string): string {
  return join(jobDir(runs), `${digest(job)}.${ext}`);
}

export function jobRunning(runs: string, job: object): boolean {
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

/** 子プロセスの段取りが終わったときに、仕事 key の終了コードを書く。 */
export function finish(runs: string, key: string, code: number): void {
  writeFileSync(join(jobDir(runs), `${key}.exit`), String(code));
}

// ---- CLI

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
  finish(runs, job, code);
  process.exitCode = code;
}

function command(runs: string, argv: string[]): void {
  const [name, ...rest] = argv;
  if (name === 'run') for (const path of rest) runSpec(runs, path);
  else if (name === 'list')
    for (const e of scan(runs)) console.log(`${e.id} ${e.kind.padEnd(8)} ${e.created} ${e.name}`);
  else console.log('run <resolved/ の下のパス>... | list');
}

if (import.meta.main) main(process.argv.slice(2));
