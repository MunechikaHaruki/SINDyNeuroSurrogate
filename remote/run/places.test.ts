import { expect, test } from 'bun:test';
import { FORMAT, type Spec } from '../../shared/spec.ts';
import { dirOf } from './places.ts';

const input = { spec: { target: 'traub-19comp', current_type: 'steady', dt: 0.01, current_params: { amplitude: 1 } }, param: null, values: [0] };
const sim = (model: { train: string; replace_targets: string[] } | null): Spec => ({ format: FORMAT, device: 'cpu', kind: 'sim', input, model });

test('specs that differ only in what they replace share the input directory but not the model', () => {
  const original = dirOf(sim(null));
  expect(original).toMatch(/^inputs\/[0-9a-f]{12}\/original$/);
  const prefix = original.slice(0, -'original'.length);
  const soma = dirOf(sim({ train: 't', replace_targets: ['soma'] }));
  const all = dirOf(sim({ train: 't', replace_targets: ['soma', 'axon'] }));
  const other = dirOf(sim({ train: 'u', replace_targets: ['soma'] }));
  for (const dir of [soma, all, other]) expect(dir.startsWith(prefix)).toBe(true);
  expect(soma).not.toBe(all);
  expect(soma).not.toBe(other);
});
