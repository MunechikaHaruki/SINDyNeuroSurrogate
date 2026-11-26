import { expect, test } from 'bun:test';
import { canonical, hashOf } from './hash.ts';
import { FORMATS, type Spec } from './spec.ts';

test('the key does not depend on the order the spec was written in', () => {
  expect(canonical({ b: 1, a: { d: [1, { f: 2, e: 3 }], c: null } })).toBe(
    '{"a":{"c":null,"d":[1,{"e":3,"f":2}]},"b":1}',
  );
  const input = {
    spec: { target: 'hh', current_type: 'steady', dt: 0.01, current_params: { a: 1, b: 2 } },
  };
  const a: Spec = { format: FORMATS.sim, device: 'cpu', kind: 'sim', input, model: null };
  const b = JSON.parse(
    `{"model":null,"input":{"spec":{"current_params":{"b":2,"a":1},"dt":0.01,"current_type":"steady","target":"hh"}},"kind":"sim","device":"cpu","format":${FORMATS.sim}}`,
  ) as Spec;
  expect(hashOf(b)).toBe(hashOf(a));
});
