// 研究コードの静的な計測（51.09）の口。開発サーバーが /api/resources/research/evaluation/metrics に載せる。
// GET のたびにこの repo の根で cloc・lizard・radon を回し、表に要る値だけを返す。受ける引数は無い。
import { fileURLToPath } from 'node:url';

const REPO = fileURLToPath(new URL('../..', import.meta.url));
const PACKAGE = 'neurosurrogate';

/** 言語ごとの行数。cloc の数え方のまま。 */
export type Language = { language: string; files: number; blank: number; comment: number; code: number };
/** 関数ごとの複雑さ。lizard の CSV の列から。file は neurosurrogate/ からのパス。 */
export type Func = { file: string; name: string; line: number; nloc: number; ccn: number; params: number };
/** ファイルごとの保守性指数。radon mi の値と A〜C の段。file は neurosurrogate/ からのパス。 */
export type Maintainability = { file: string; mi: number; rank: string };
export type Metrics = {
  languages: Language[];
  functions: Func[];
  maintainability: Maintainability[];
  averageCc: number;
};

/** SINDy の repo でコマンドを回し、標準出力を返す。落ちたら標準エラーを投げる。 */
function run(command: string[]): string {
  const { exitCode, stdout, stderr } = Bun.spawnSync(command, { cwd: REPO });
  if (exitCode !== 0) throw new Error(`${command.join(' ')}: ${stderr.toString()}`);
  return stdout.toString();
}

/** cloc --vcs=git の JSON から言語の行を取り出す。header と SUM は除く。 */
function languages(): Language[] {
  const json = JSON.parse(run(['cloc', '.', '--vcs=git', '--json', '--quiet'])) as Record<
    string,
    Record<string, number>
  >;
  const rows: Language[] = [];
  for (const language of Object.keys(json)) {
    if (language === 'header' || language === 'SUM') continue;
    const row = json[language];
    rows.push({ language, files: row.nFiles, blank: row.blank, comment: row.comment, code: row.code });
  }
  return rows;
}

/** CSV の一つの欄から囲みの引用符を外す。 */
function unquote(cell: string): string {
  return cell.replace(/^"|"$/g, '');
}

/** lizard --csv の行。列は nloc, ccn, token, params, length, location, file, name, long_name, start, end。 */
function functions(): Func[] {
  const rows: Func[] = [];
  for (const line of run(['uv', 'run', 'lizard', `./${PACKAGE}`, '--csv'])
    .trim()
    .split('\n')) {
    const cells = line.match(/("[^"]*"|[^,]+)/g) ?? [];
    rows.push({
      file: unquote(cells[6]).replace(`./${PACKAGE}/`, ''),
      name: unquote(cells[7]),
      line: Number(cells[9]),
      nloc: Number(cells[0]),
      ccn: Number(cells[1]),
      params: Number(cells[3]),
    });
  }
  return rows;
}

/** radon mi -j。ファイルごとの保守性指数（0〜100、高いほど保守しやすい）。 */
function maintainability(): Maintainability[] {
  const json = JSON.parse(run(['uv', 'run', 'radon', 'mi', `./${PACKAGE}`, '-j'])) as Record<
    string,
    { mi: number; rank: string }
  >;
  const rows: Maintainability[] = [];
  for (const file of Object.keys(json)) rows.push({ file: file.replace(`${PACKAGE}/`, ''), mi: json[file].mi, rank: json[file].rank });
  return rows;
}

/** radon cc -j の全ブロックの循環的複雑度の平均。radon cc -a が出す値。 */
function averageCc(): number {
  const json = JSON.parse(run(['uv', 'run', 'radon', 'cc', `./${PACKAGE}`, '-j'])) as Record<
    string,
    { complexity: number }[]
  >;
  let sum = 0;
  let count = 0;
  for (const file of Object.keys(json)) {
    for (const block of json[file]) {
      sum += block.complexity;
      count += 1;
    }
  }
  return count === 0 ? 0 : sum / count;
}

/** 三つの道具を回した結果。 */
export function GET(): Metrics {
  return { languages: languages(), functions: functions(), maintainability: maintainability(), averageCc: averageCc() };
}
