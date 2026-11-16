import { expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { BENCH, type BenchOwn } from '../../shared/runs.ts';
import { benchRows, ncuKernels, ncuSummed, smiTotalBytes, smiUsedMib } from './bench.ts';
import { runFiles } from './waves.ts';

/** ncu の CSV の 1 行。 */
function line(id: number, metric: string, value: string): string {
  return `"${id}","4242","python","127.0.0.1","step_kernel","1","7","(1, 1, 1)","(1024, 1, 1)","0","8.9","${metric}","byte","${value}"`;
}

const HEADER =
  '"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","Metric Name","Metric Unit","Metric Value"';

/** 区画 2 つ（カーネル 2 本）の ncu の標準出力。==PROF== の行と、値の無い行を混ぜる。 */
const NCU = [
  '==PROF== Connected to process 4242',
  HEADER,
  line(0, 'launch__registers_per_thread', '108'),
  line(0, 'l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum', '1,000'),
  line(0, 'l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum', '0'),
  line(0, 'l1tex__t_sector_hit_rate.pct', '10'),
  line(0, 'dram__bytes_read.sum', 'n/a'),
  line(1, 'launch__registers_per_thread', '120'),
  line(1, 'l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum', '2000'),
  line(1, 'l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum', '1000'),
  line(1, 'l1tex__t_sector_hit_rate.pct', '50'),
  line(1, 'dram__bytes_read.sum', '64'),
  '==PROF== Disconnected from process 4242',
].join('\n');

test('ncu kernels add up amounts, take the most registers and weight hit rates by global traffic', () => {
  const kernels = ncuKernels(NCU);
  expect(Object.keys(kernels)).toEqual(['0', '1']);
  const summed = ncuSummed(kernels);
  expect(summed.registers_per_thread).toBe(120);
  expect(summed.global_load_bytes).toBe(3000);
  expect(summed.global_store_bytes).toBe(1000);
  // 重みは 1000 と 3000
  expect(summed.l1_hit_pct).toBeCloseTo((10 * 1000 + 50 * 3000) / 4000);
  // カーネル 0 の値が n/a なので決まらない。測っていない指標も決まらない
  expect(summed.dram_read_bytes).toBeNull();
  expect(summed.l2_hit_pct).toBeNull();
});

test('nvidia-smi rows are picked by pid and the total is read in MiB', () => {
  const apps = 'pid, used_gpu_memory [MiB]\n4100, 512 MiB\n4242, 226 MiB\n';
  expect(smiUsedMib(apps, 4242)).toBe(226);
  expect(smiUsedMib(apps, 1)).toBeNull();
  expect(smiTotalBytes('memory.total [MiB]\n16376 MiB\n')).toBe(16376 * 1024 * 1024);
  expect(smiTotalBytes(null)).toBeNull();
});

test('a bench run is gathered per cell count from the tool outputs, in cell order', () => {
  const runs = mkdtempSync(join(tmpdir(), 'bench-'));
  const run = 'bench/abc';
  for (const cells of [1048576, 1024]) {
    const dir = join(runs, run, String(cells));
    mkdirSync(dir, { recursive: true });
    const own: BenchOwn = {
      cells,
      compartments: 2,
      steps: 128,
      compile_s: 0.4,
      wall_s: 0.5,
      comp_states: { soma: ['V', 'z1'] },
      decoded: { soma: ['H', 'N'] },
      vram_state_bytes: 8 * cells,
      expected_step_load_bytes: 8 * cells,
      expected_step_store_bytes: 8 * cells,
      spike_counts: [[0]],
      pid: 4242,
    };
    writeFileSync(join(dir, BENCH.own), JSON.stringify(own));
    writeFileSync(join(dir, BENCH.jaxPlaced), JSON.stringify({ bytes_in_use: 100 }));
    writeFileSync(join(dir, BENCH.jaxAfter), JSON.stringify({ bytes_in_use: 100, peak_bytes_in_use: 400 }));
    writeFileSync(join(dir, BENCH.smiApps), 'pid, used_gpu_memory [MiB]\n4242, 226 MiB\n');
    if (cells === 1024) writeFileSync(join(dir, BENCH.ncu), NCU);
  }
  const rows = benchRows(runFiles(runs), run);
  expect(rows.map(r => r.cells)).toEqual([1024, 1048576]);
  expect(rows[0]).toMatchObject({
    input_bytes: 100,
    peak_bytes: 400,
    work_bytes: 300,
    smi_used_mib: 226,
    vram_total_bytes: null,
    registers_per_thread: 120,
  });
  expect(rows[0].cell_steps_per_s).toBeCloseTo((1024 * 128) / 0.5);
  expect('pid' in rows[0]).toBe(false);
  expect(rows[1].registers_per_thread).toBeUndefined();
});
