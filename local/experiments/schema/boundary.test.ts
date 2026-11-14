import { expect, test } from 'bun:test';
import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = fileURLToPath(new URL('../../..', import.meta.url));

/** 調べるディレクトリと、そこから import してよい場所。手元（local/）と計算する機械（remote/）は互いのコードを import せず、両方が使うのは shared/ だけ。 */
const LOCAL = ['local/', 'shared/'];
const ALLOWED: Record<string, string[]> = {
  local: LOCAL,
  remote: ['remote/', 'shared/'],
  'local/experiments/schema': LOCAL,
  'remote/run': ['remote/', 'shared/'],
  'remote/read': ['remote/', 'shared/'],
  shared: ['shared/'],
};

/** ディレクトリの直下の .ts と、その中の相対 import の行き先（repo からのパス）。 */
function imports(dir: string): { file: string; target: string }[] {
  const found: { file: string; target: string }[] = [];
  for (const name of readdirSync(join(REPO, dir))) {
    if (!name.endsWith('.ts')) continue;
    const file = join(REPO, dir, name);
    for (const [, path] of readFileSync(file, 'utf8').matchAll(/from '(\.[^']+)'/g))
      found.push({ file: relative(REPO, file), target: relative(REPO, resolve(dirname(file), path)) });
  }
  return found;
}

/** 側の下の、.ts を直に置くディレクトリを深さを問わず（node_modules、Python の置き場、名前に . を含む生成物を除く）。 */
function tsDirs(side: string): string[] {
  const found: string[] = [];
  for (const e of readdirSync(join(REPO, side), { withFileTypes: true })) {
    if (!e.isDirectory() || ['node_modules', 'neurosurrogate', 'tests'].includes(e.name) || e.name.includes('.')) continue;
    const dir = `${side}/${e.name}`;
    if (readdirSync(join(REPO, dir)).some(name => name.endsWith('.ts'))) found.push(dir);
    found.push(...tsDirs(dir));
  }
  return found;
}

test('every TS directory of local/ and remote/ is checked', () => {
  expect([...tsDirs('local'), ...tsDirs('remote')].filter(d => !(d in ALLOWED))).toEqual([]);
});

test('local/ and remote/ share only shared/, so only the resolved specs and the API JSON cross between them', () => {
  const crossing: string[] = [];
  for (const [dir, allowed] of Object.entries(ALLOWED))
    for (const { file, target } of imports(dir))
      if (!allowed.some(prefix => target.startsWith(prefix))) crossing.push(`${file} → ${target}`);
  expect(crossing).toEqual([]);
});
