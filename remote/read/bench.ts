// GPU の実測（runs/bench/<実測>/）の答えと集計。run の <細胞の数>/ に置いたツールごとの出力（bench.py が測ったもの、JAX の memory_stats、nvidia-smi と ncu の CSV）を読み、細胞の数ごとの 1 行（BenchRow）に集める。
// run は生の出力だけを持ち、ここが頼まれるたびに計算する。ncu の区画ごとのカーネルの合わせ方（量は和、レジスタは最大、当たり率は global の読み書きの量で重み付け）もここが持つ。
import type { BenchReply, BenchRequest } from '../../shared/api.ts';
import { BENCH, type BenchOwn, type BenchRow } from '../../shared/runs.ts';
import type { RunEnv } from '../../shared/spec.ts';
import { specAt } from '../run/places.ts';
import { experimentStatus } from '../run/status.ts';
import { type Context, planOf, plans } from './experiments.ts';
import type { RunFiles } from './waves.ts';

/** ncu の指標の名前 → 行の欄。 */
const NCU_KEYS = {
  launch__registers_per_thread: 'registers_per_thread',
  launch__thread_count: 'threads',
  'l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum': 'global_store_bytes',
  'l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum': 'global_load_bytes',
  'l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum': 'local_store_bytes',
  'l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum': 'local_load_bytes',
  'dram__bytes_write.sum': 'dram_write_bytes',
  'dram__bytes_read.sum': 'dram_read_bytes',
  'l1tex__t_sector_hit_rate.pct': 'l1_hit_pct',
  'lts__t_sector_hit_rate.pct': 'l2_hit_pct',
} as const;

/** 当たり率の重みに使う、global の読み込みと書き込みの量の指標。 */
const NCU_LOAD = 'l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum';
const NCU_STORE = 'l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum';

type NcuKey = (typeof NCU_KEYS)[keyof typeof NCU_KEYS];

/** CSV の 1 行の、引用符で囲んだ欄。 */
function quoted(line: string): string[] {
  const fields: string[] = [];
  for (const match of line.matchAll(/"((?:[^"]|"")*)"/g)) fields.push(match[1].replaceAll('""', '"'));
  return fields;
}

/** ncu の CSV（標準出力のまま）を、カーネルの ID → 指標の名前 → 値 にする。CSV でない行（==PROF== など）と値の無い行は飛ばす。 */
export function ncuKernels(csv: string): Record<string, Record<string, number>> {
  const lines: string[] = [];
  for (const line of csv.split('\n')) if (line.startsWith('"')) lines.push(line);
  const kernels: Record<string, Record<string, number>> = {};
  if (lines.length === 0) return kernels;
  const header = quoted(lines[0]);
  const id = header.indexOf('ID');
  const name = header.indexOf('Metric Name');
  const value = header.indexOf('Metric Value');
  for (const line of lines.slice(1)) {
    const fields = quoted(line);
    const number = Number(fields[value]?.replaceAll(',', ''));
    if (fields[value] === undefined || fields[value] === '' || !Number.isFinite(number)) continue;
    kernels[fields[id]] ??= {};
    kernels[fields[id]][fields[name]] = number;
  }
  return kernels;
}

/** 1 刻みの区画ごとのカーネルを合わせる。量は和、レジスタは最大、当たり率は global の読み書きの量で重み付け。どれかのカーネルに無い指標は null。 */
export function ncuSummed(kernels: Record<string, Record<string, number>>): Record<NcuKey, number | null> {
  const rows = Object.values(kernels);
  const out = {} as Record<NcuKey, number | null>;
  for (const [metric, key] of Object.entries(NCU_KEYS) as [string, NcuKey][]) {
    const values: number[] = [];
    for (const row of rows) if (row[metric] !== undefined) values.push(row[metric]);
    if (values.length !== rows.length || rows.length === 0) {
      out[key] = null;
      continue;
    }
    if (key === 'registers_per_thread') out[key] = Math.max(...values);
    else if (key.endsWith('_pct')) {
      let total = 0;
      let weighted = 0;
      for (let i = 0; i < rows.length; i++) {
        const weight = (rows[i][NCU_LOAD] ?? 0) + (rows[i][NCU_STORE] ?? 0);
        total += weight;
        weighted += values[i] * weight;
      }
      out[key] = total > 0 ? weighted / total : null;
    } else {
      let sum = 0;
      for (const v of values) sum += v;
      out[key] = sum;
    }
  }
  return out;
}

/** nvidia-smi の CSV（見出しと単位つき）の、見出しの次からの行の欄。 */
function smiRows(csv: string): string[][] {
  const rows: string[][] = [];
  for (const line of csv.trim().split('\n').slice(1)) rows.push(line.split(',').map(field => field.trim()));
  return rows;
}

/** 欄の頭の数（「226 MiB」の 226）。 */
function leading(field: string | undefined): number | null {
  const value = Number.parseFloat(field ?? '');
  return Number.isFinite(value) ? value : null;
}

/** --query-compute-apps=pid,used_memory の CSV から、プロセス pid の使用量 [MiB]。無ければ null。 */
export function smiUsedMib(csv: string | null, pid: number): number | null {
  if (csv === null) return null;
  for (const row of smiRows(csv)) if (leading(row[0]) === pid) return leading(row[1]);
  return null;
}

/** --query-gpu=memory.total の CSV から、GPU の VRAM の全容量 [バイト]。無ければ null。 */
export function smiTotalBytes(csv: string | null): number | null {
  if (csv === null) return null;
  const mib = leading(smiRows(csv)[0]?.[0]);
  return mib === null ? null : mib * 1024 * 1024;
}

/** 細胞の数 1 つのツールの出力を、1 行に集める。 */
export function benchRow(
  own: BenchOwn,
  placed: Record<string, number> | null,
  after: Record<string, number> | null,
  smiApps: string | null,
  smiGpu: string | null,
  ncu: string | null,
): BenchRow {
  const { pid, ...measured } = own;
  const input = placed?.bytes_in_use ?? null;
  const peak = after?.peak_bytes_in_use ?? null;
  const row: BenchRow = {
    ...measured,
    cell_steps_per_s: (own.cells * own.steps) / own.wall_s,
    peak_bytes: peak,
    input_bytes: input,
    work_bytes: peak === null || input === null ? null : peak - input,
    smi_used_mib: smiUsedMib(smiApps, pid),
    vram_total_bytes: smiTotalBytes(smiGpu),
  };
  return ncu === null ? row : { ...row, ...ncuSummed(ncuKernels(ncu)) };
}

/** GPU の実測の run の、細胞の数ごとの行。並びは細胞の数の順。 */
export function benchRows(files: RunFiles, run: string): BenchRow[] {
  const counts: number[] = [];
  for (const name of files.dirs(run)) if (/^\d+$/.test(name)) counts.push(Number(name));
  counts.sort((a, b) => a - b);
  const rows: BenchRow[] = [];
  for (const cells of counts) {
    const read = (file: string) => files.text(run, BENCH.at(cells, file));
    const own = read(BENCH.own);
    if (own === null) throw new Error(`${run}/${BENCH.at(cells, BENCH.own)} が無い`);
    const placed = read(BENCH.jaxPlaced);
    const after = read(BENCH.jaxAfter);
    rows.push(
      benchRow(
        JSON.parse(own) as BenchOwn,
        placed === null ? null : JSON.parse(placed),
        after === null ? null : JSON.parse(after),
        read(BENCH.smiApps),
        read(BENCH.smiGpu),
        read(BENCH.ncu),
      ),
    );
  }
  return rows;
}

/** 細胞の数ごとのファイルの名前 → それを書いたツール。 */
const TOOLS: Record<string, string> = {
  [BENCH.own]: 'bench.py',
  [BENCH.jaxPlaced]: 'JAX',
  [BENCH.jaxAfter]: 'JAX',
  [BENCH.smiApps]: 'nvidia-smi',
  [BENCH.smiGpu]: 'nvidia-smi',
  [BENCH.ncu]: 'ncu',
};

/** GPU の実測 1 本。まだ回していなければ行は空。files なら細胞の数ごとのツールの出力そのものも返す。 */
export function bench(ctx: Context, request: BenchRequest): BenchReply {
  const plan = planOf(plans(), request.experiment);
  const declared = plan.benches.find(b => b.name === request.bench);
  if (!declared) throw new Error(`${plan.id} に GPU の実測 ${request.bench} が無い`);
  const found = experimentStatus(ctx.runs, plan.id).benches.find(b => b.bench === declared.name)!;
  const done = found.state === 'done';
  const spec = specAt(found.run.spec);
  const meta = done ? ctx.files.json<{ env: RunEnv; created: string }>(found.run.id, 'meta.json') : null;
  const reply: BenchReply = {
    name: declared.name,
    input: declared.series,
    model: declared.model === 'original' ? null : declared.model,
    cells: declared.cells,
    ncu: declared.ncu,
    dt: spec.kind === 'bench' ? spec.input.spec.dt : 0.01,
    run: { state: found.state, spec: found.run.spec, run: found.run.id },
    machine: meta && { device: meta.env.device, host: meta.env.host, created: meta.created },
    rows: done ? benchRows(ctx.files, found.run.id) : [],
    tools: TOOLS,
  };
  if (!request.files || !done) return reply;
  const files: Record<number, Record<string, string | null>> = {};
  for (const cells of declared.cells)
    files[cells] = Object.fromEntries(Object.keys(TOOLS).map(file => [file, ctx.files.text(found.run.id, BENCH.at(cells, file))]));
  return { ...reply, files };
}
