// 置き場。手元から送られてきた resolved/（手元の local/resolve.ts が書く）からパスで spec を引き、run を置く runs/ では、spec から置き場を決め
// （学習は models/<ハッシュ>/、入力 × モデルは inputs/<入力のハッシュ>/<original かモデルの鍵>/、GPU の実測は bench/<ハッシュ>/）、完成した run を並べ、run 1 本の状態（済み・計算中・置換できない・まだ）を読む。
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { IndexEntry, Spec, State } from '../../shared/spec.ts';
import { digest, hashOf } from '../../shared/hash.ts';

/** 送られてきた resolved/ の置き場。手元が組んで送る（local/remote.sh）ので、どこにあるかは SINDY_RESOLVED で受け取る
 * （計算する機械のイメージは /code/resolved、手元で回すときは local/experiments/resolved）。 */
export function resolvedDir(): string {
  const dir = process.env.SINDY_RESOLVED;
  if (!dir)
    throw new Error('SINDY_RESOLVED に送られてきた resolved/ の場所を入れる（手元なら local/experiments/resolved）');
  return resolve(dir);
}

/** resolved/ の下のパスの spec.json。resolved/ の外を指すパスは弾く。 */
export function specAt(path: string): Spec {
  const dir = resolvedDir();
  const file = resolve(dir, path);
  if (!file.startsWith(`${dir}/`) || !existsSync(file)) throw new Error(`resolved/${path} が無い`);
  return JSON.parse(readFileSync(file, 'utf8')) as Spec;
}

/** 計算する機械の側（Python の project）。Python はここで起こす。 */
export const PROJECT = fileURLToPath(new URL('..', import.meta.url));

/** Python を起こすコマンド。計算する機械のイメージは venv の python を SINDY_PYTHON で渡す。 */
export const PYTHON = process.env.SINDY_PYTHON?.split(' ') ?? ['uv', 'run', 'python'];

/** 既定の run の置き場。計算する機械のイメージは SINDY_RUNS で載せた場所を指す。 */
export const RUNS = process.env.SINDY_RUNS ?? join(fileURLToPath(new URL('../..', import.meta.url)), 'runs');

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

/** 置換できないと分かった組の印。置換できるかは Python だけが知るので、一度聞いた答えを置き場の隣に覚えておく。 */
export function unreplaceableMark(runs: string, id: string): string {
  return join(runs, `${id}.unreplaceable`);
}

/** pid のプロセスが生きているか。 */
export function alive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

/** 書きかけの run（<ハッシュ>.partial）を、書いているプロセスが生きているか。run.py が中に pid を置く。 */
export function running(partial: string): boolean {
  const file = join(partial, 'pid');
  return existsSync(file) && alive(Number(readFileSync(file, 'utf8')));
}

/** run 1 本の状態。 */
export function runState(runs: string, id: string): State {
  if (existsSync(unreplaceableMark(runs, id))) return 'unreplaceable';
  if (existsSync(join(runs, id))) return 'done';
  return running(join(runs, `${id}.partial`)) ? 'running' : 'todo';
}
