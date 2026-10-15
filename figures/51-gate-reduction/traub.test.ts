import { expect, test } from 'bun:test';
import fixture from './traub.reference.json';
import * as model from './traub.ts';

const recovery: model.Grouping = { phi: [], psi: [['h', 'n']] };
const threeVariables: model.Grouping = {
  phi: ['m'],
  psi: [
    ['h', 'n'],
    ['a', 'c'],
  ],
};
/** 標本の膜電位の列。 */
function voltages(samples: model.Sample[], grouping?: model.Grouping): number[] {
  const values: number[] = [];
  for (const sample of samples) values.push(model.readSample(sample, grouping).v);
  return values;
}
/** 標本の Ca 状態 χ の列。 */
function calcium(samples: model.Sample[], grouping?: model.Grouping): number[] {
  const values: number[] = [];
  for (const sample of samples) values.push(model.readSample(sample, grouping).chi);
  return values;
}

test('Traub rates and analytic slopes agree with the research implementation (JAX float64)', () => {
  for (const row of fixture.rates) {
    const actual = model.rates(row.v);
    // 参照値はゲートの並び（m,h,n,a,b,s,r,c）で [α, β] の組として入っている。
    row.rates.forEach((pair, i) => {
      const rate = actual[model.gateNames[i]];
      expect(rate.alpha).toBeCloseTo(pair[0], 11);
      expect(rate.beta).toBeCloseTo(pair[1], 11);
    });
    const r = model.recoverySteady(row.v);
    expect(r.slopeH).toBeCloseTo(row.slopes[0], 11);
    expect(r.slopeN).toBeCloseTo(row.slopes[1], 11);
  }
});

test('all 11 derivatives, including calcium saturation, match the research implementation', () => {
  const actual = model.fullDerivative(fixture.rhs.state, fixture.rhs.input);
  fixture.rhs.expected.forEach((value, i) => expect(actual[i]).toBeCloseTo(value, 10));
});

test('compatible initialization preserves all retained derivatives and gives zero recovery velocity', () => {
  const full = model.initial(),
    reduced = model.initial(recovery);
  expect(full).toHaveLength(11);
  expect(reduced).toHaveLength(10);
  // h,n の群は m の次に入るので、並びは (V, m, ψ, a, b, s, r, c, q, χ)。
  expect(model.groupLayout(recovery)).toEqual([['m'], ['h', 'n'], ['a'], ['b'], ['s'], ['r'], ['c']]);
  const df = model.fullDerivative(full, 3),
    dr = model.reducedDerivative(recovery, reduced, 3);
  expect(dr[2]).toBeCloseTo(0, 12);
  [0, 1, 4, 5, 6, 7, 8, 9, 10].forEach((j, i) => expect(df[j]).toBeCloseTo(dr[i < 2 ? i : i + 1], 12));
});

test('default full and reduced trajectories converge when tolerances and maximum step are tightened', () => {
  for (const grouping of [undefined, recovery, threeVariables]) {
    const normal = model.simulate({ mean: 5, amplitude: 5 }, grouping);
    const fine = model.simulate({ mean: 5, amplitude: 5 }, grouping, 1e-9, 0.05);
    expect(normal).toHaveLength(5001);
    expect(normal.at(-1)!.time).toBe(500);
    expect(model.rmse(voltages(normal, grouping), voltages(fine, grouping))).toBeLessThan(0.001);
    expect(model.rmse(calcium(normal, grouping), calcium(fine, grouping))).toBeLessThan(0.001);
    const a = model.spikes(normal),
      b = model.spikes(fine);
    expect(a.length).toBeGreaterThan(0);
    expect(a.length).toBe(b.length);
    a.forEach((time, i) => expect(Math.abs(time - b[i])).toBeLessThan(0.001));
  }
});

test('zero input stays silent and does not introduce a spurious reduced spike', () => {
  const a = model.simulate({ mean: 0, amplitude: 0 });
  for (const grouping of [recovery, threeVariables]) {
    const b = model.simulate({ mean: 0, amplitude: 0 }, grouping);
    expect(model.spikes(a)).toEqual([]);
    expect(model.spikes(b)).toEqual([]);
    // φ 群の m は −60 mV からの落ち着き方を変えるので、無入力の 200 ms 後にわずかな差が残る。
    expect(model.rmse(voltages(a), voltages(b, grouping))).toBeLessThan(0.001);
  }
});

test('invalid recovery weights stop the calculation instead of silently clipping', () => {
  const state = model.initial(recovery);
  state[0] = -76;
  expect(() => model.reducedDerivative(recovery, state, 0)).toThrow('正の重みの適用範囲外');
});

test('the general reduced right-hand side reproduces the h,n equation of 51.01-apply-kam', () => {
  // 本文「h,n の 10 状態系」の dψ/dt = α_h f_h + α_n f_n を、解析的な傾きで別に組み立てて比べる。
  const y = model.initial(recovery);
  y[0] = -45;
  y[2] = -58;
  const x = model.readReduced(recovery, y);
  const rate = model.rates(-45);
  const eq = model.recoverySteady(-58);
  const dh = 30 * x.gates.m * x.gates.m * (-45 - 55) * eq.slopeH;
  const dn = 15 * (-45 + 75) * eq.slopeN;
  const fh = (model.speed(rate.h) * (model.steady(rate.h) - eq.h)) / eq.slopeH;
  const fn = (model.speed(rate.n) * (model.steady(rate.n) - eq.n)) / eq.slopeN;
  const expected = (dh * fh + dn * fn) / (dh + dn);
  expect(model.reducedDerivative(recovery, y, 0)[2] / expected).toBeCloseTo(1, 6);
});

test('the three-variable grouping keeps V, three gate variables, s, r and the calcium system', () => {
  expect(model.groupLayout(threeVariables)).toEqual([['h', 'n'], ['a', 'c'], ['b'], ['s'], ['r']]);
  expect(model.initial(threeVariables)).toHaveLength(8);
  // φ 群の m は φ = V での定常値から作り直す。
  const y = model.initial(threeVariables);
  y[0] = -40;
  expect(model.readReduced(threeVariables, y).gates.m).toBe(model.steady(model.rates(-40).m));
  // 同じ入力で、発火数は元モデルと ±1 以内。
  const config = { mean: 5, amplitude: 5 };
  const full = model.spikes(model.simulate(config)).length;
  const reduced = model.spikes(model.simulate(config, threeVariables)).length;
  expect(Math.abs(full - reduced)).toBeLessThanOrEqual(1);
});

test('weights and consistency conditions follow equations (6) and (10) for a general group', () => {
  expect(model.recoveryWeights([1, 3])).toEqual([0.25, 0.75]);
  // 速さがそろえば式(10)の左辺は 0。
  model.speedConsistency([0.2, 0.3, 0.5], [2, 2, 2]).forEach(value => expect(value).toBe(0));
  // α = (0.25, 0.75)、k = (4, 1) なら Σαk = 1.75、Σk = 5。
  const [h, n] = model.speedConsistency([0.25, 0.75], [4, 1]);
  expect(h).toBeCloseTo(2.25 / 5, 12);
  expect(n).toBeCloseTo(0.75 / 5, 12);
});

test('consistency criteria keep their reference values at -60, -40 and 0 mV', () => {
  const close = (actual: number, expected: number) => expect(Math.abs(actual / expected - 1)).toBeLessThan(1e-3);
  // [V, D_h, D_n, k_h, k_n, 式(10)左辺の大きい方]
  const rows = [
    [-60, 0.000745, 0.05381, 0.3305, 0.4127, 0.1091],
    [-40, 17.5428, 3.9941, 0.1803, 0.2624, 0.1511],
    [0, 0.286, 10.8856, 3.9398, 0.4931, 0.7576],
  ];
  for (const [v, dh, dn, kh, kn, speed] of rows) {
    const c = model.criteria(v);
    [
      [c.dh, dh],
      [c.dn, dn],
      [c.kh, kh],
      [c.kn, kn],
      [Math.max(c.speedH, c.speedN), speed],
    ].forEach(([a, e]) => close(a, e));
  }
});

test('equilibria are steady states of the full model at that constant current', () => {
  for (const current of [0, 2, 5]) {
    const roots = model.equilibria(current);
    expect(roots.length).toBeGreaterThan(0);
    for (const v of roots)
      model.fullDerivative(model.steadyState(v), current).forEach(d => expect(Math.abs(d)).toBeLessThan(1e-9));
  }
});

test('the steady I-V curve is N-shaped and gives the equilibria quoted in 51.01-apply-kam', () => {
  const roots = model.equilibria(0);
  expect(roots).toHaveLength(3);
  [-69, -58, -31].forEach((v, i) => expect(Math.abs(roots[i] - v)).toBeLessThan(0.5));
  // 静止電位と閾値の間にある極大。I₀ がこれを超えると根は一つになる。
  let peak = -Infinity;
  for (let v = roots[0]; v <= roots[1]; v += 0.05) peak = Math.max(peak, model.steadyCurrent(v));
  expect(Math.abs(peak - 0.87)).toBeLessThan(0.005);
  expect(model.equilibria(0.9)).toHaveLength(1);
});

test('weights stay strictly between 0 and 1 inside the reversal potentials', () => {
  for (let v = -74.5; v <= 54.5; v += 0.5) {
    const c = model.criteria(v);
    expect(c.alphaH).toBeGreaterThan(0);
    expect(c.alphaH).toBeLessThan(1);
  }
});

test('equivalent potentials invert the steady-state curves and are NaN where they are undefined', () => {
  for (const name of ['m', 'h', 'n', 'a', 'b'] as const) {
    for (let v = -100; v <= 50; v += 10) {
      expect(model.equivalentPotential(name, model.steady(model.rates(v)[name]))).toBeCloseTo(v, 6);
    }
  }
  // c̄ は −10 mV より上で 1 に張り付くので、逆関数は −10 mV 未満だけで取る。
  for (let v = -100; v < -10; v += 10) {
    expect(model.equivalentPotential('c', model.steady(model.rates(v).c))).toBeCloseTo(v, 6);
  }
  expect(model.equivalentPotential('n', 1)).toBeNaN();
});

test('the general group criteria reproduce the h,n criteria at every voltage', () => {
  for (let v = -74.5; v <= 54.5; v += 0.5) {
    const old = model.criteria(v);
    const general = model.groupCriteria(['h', 'n'], v);
    expect(general.weights[0]).toBeCloseTo(old.alphaH, 6);
    expect(general.consistency[0]).toBeCloseTo(old.speedH, 6);
    expect(general.consistency[1]).toBeCloseTo(old.speedN, 6);
  }
});

test('group conditions on the full trajectory accept same-sign groups and reject mixed signs', () => {
  const samples = model.simulate({ mean: 5, amplitude: 5 });
  const total = samples.length * 0.1;
  // h,n はどちらも D > 0、m は D < 0 で φ に入れられる。h,b は D の符号が逆。
  expect(model.checkTrajectory(['h', 'n'], samples, false).durations.ok).toBeCloseTo(total, 6);
  expect(model.checkTrajectory(['m'], samples, true).durations.ok).toBeCloseTo(total, 6);
  expect(model.checkTrajectory(['h', 'b'], samples, false).durations.wrongSign).toBeCloseTo(total, 6);
  expect(model.checkTrajectory(['n'], samples, true).durations.wrongSign).toBeCloseTo(total, 6);
  for (const check of model.checkTrajectory(['h', 'n'], samples, false).checks) {
    check.weights.forEach(alpha => {
      expect(alpha).toBeGreaterThan(0);
      expect(alpha).toBeLessThan(1);
    });
  }
});

test('group conditions report a missing equivalent potential and a group with no sensitivity', () => {
  const state = model.readFull(model.steadyState(-60));
  // n̄ は 1 に届かないので、n = 1 は等価電位を持たない（条件 1）。
  expect(model.checkGroup(['h', 'n'], { ...state, gates: { ...state.gates, n: 1 } }, false).verdict).toBe(
    'noEquivalent',
  );
  // χ = 0 では C 電流が流れず D_c = 0 なので、c だけの群は重みが決まらない。
  expect(model.checkGroup(['c'], { ...state, chi: 0 }, false).verdict).toBe('noWeight');
});

test('steady-state curves are monotonic as 51.01-apply-kam states, except the flat ranges of r and c', () => {
  // 本文 4 ページ目：m,n,a,s は増加関数、h,b は減少関数で、全電位で x̄' ≠ 0。
  for (let v = -100; v <= 60; v += 0.5) {
    for (const name of ['m', 'n', 'a', 's'] as const) expect(model.steadySlope(name, v)).toBeGreaterThan(0);
    for (const name of ['h', 'b'] as const) expect(model.steadySlope(name, v)).toBeLessThan(0);
  }
  // r̄ は −60 mV 以下で、c̄ は −10 mV より上で 1 に張り付き、傾きが 0 になる。
  for (let v = -100; v <= -60.5; v += 0.5) {
    expect(model.steady(model.rates(v).r)).toBe(1);
    expect(model.steadySlope('r', v)).toBe(0);
  }
  for (let v = -9.5; v <= 60; v += 0.5) {
    expect(model.steady(model.rates(v).c)).toBe(1);
    expect(model.steadySlope('c', v)).toBe(0);
  }
  expect(model.flatRanges('r', -100, 60, 0.1)).toEqual([{ low: -100, high: -60 }]);
  expect(model.flatRanges('c', -100, 60, 0.1)).toEqual([{ low: -9.9, high: 60 }]);
  for (const name of ['m', 'h', 'n', 'a', 'b', 's'] as const) expect(model.flatRanges(name, -100, 60, 0.1)).toEqual([]);
  // 静止電位では r が定常値 1 にいるので、等価電位が一つに決まらない。
  expect(model.readFull(model.steadyState(-69)).gates.r).toBe(1);
});

test('sensitivity sign bands depend only on the equivalent potential, with zeros in the flat ranges of r and c', () => {
  // D の符号は sign x̄'(v)·sign(V − E_x)。E_K < V < E_Na では、m,s,b が負、h,n,r,a,c が正（平らな範囲は 0）。
  expect(model.driveSign('m')).toBe(-1);
  expect(model.driveSign('s')).toBe(-1);
  expect(model.driveSign('n')).toBe(1);
  expect(model.driveSign('c')).toBe(1);
  for (const name of ['m', 's', 'b'] as const)
    expect(model.sensitivitySignBands(name, -100, 60, 0.5)).toEqual([{ low: -100, high: 60, sign: -1 }]);
  for (const name of ['h', 'n', 'a'] as const)
    expect(model.sensitivitySignBands(name, -100, 60, 0.5)).toEqual([{ low: -100, high: 60, sign: 1 }]);
  expect(model.sensitivitySignBands('r', -100, 60, 0.5)).toEqual([
    { low: -100, high: -60, sign: 0 },
    { low: -60, high: 60, sign: 1 },
  ]);
  expect(model.sensitivitySignBands('c', -100, 60, 0.5)).toEqual([
    { low: -100, high: -10, sign: 1 },
    { low: -10, high: 60, sign: 0 },
  ]);
});

test('alpha0 from Kepler eq (9) is 1 for an empty phi group and matches the quadratic root for one gate', () => {
  expect(model.phiWeight(2, [], [])).toBe(1);
  const d0 = 2;
  const dm = -1.5;
  for (const k of [1, 10, 1000]) {
    const ck = 3 * k;
    const closed = (2 * ck) / (ck + d0 + Math.sqrt((ck - d0) ** 2 - 4 * ck * dm));
    expect(model.phiWeight(d0, [dm], [k])).toBeCloseTo(closed, 10);
  }
  expect(model.phiWeight(d0, [dm], [1e6])).toBeCloseTo(1, 5);
  // 閾値付近のように D_m が D₀ を上回って ΣD < 0 でも、同じ二次式の根を選ぶ。
  const strong = -30;
  const ck = 3 * 8;
  const closedStrong = (2 * ck) / (ck + d0 + Math.sqrt((ck - d0) ** 2 - 4 * ck * strong));
  expect(model.phiWeight(d0, [strong], [8])).toBeCloseTo(closedStrong, 10);
});

test('alpha0 from Kepler eq (9) satisfies the equation for two phi gates', () => {
  const d0 = 2;
  const sensitivities = [-1.5, -0.4];
  const speeds = [5, 0.8];
  const alpha0 = model.phiWeight(d0, sensitivities, speeds);
  const total = d0 + sensitivities[0] + sensitivities[1];
  let residual = alpha0 * total - d0;
  for (let i = 0; i < 2; i++) residual -= (3 * speeds[i] * sensitivities[i]) / (3 * speeds[i] - alpha0 * total);
  expect(Math.abs(residual)).toBeLessThan(1e-9);
});

test('solving alpha0 changes the reduced model only through the membrane equation', () => {
  const y = model.initial(threeVariables);
  y[0] = -45;
  const one = model.reducedDerivative(threeVariables, y, 5);
  const solved = model.reducedDerivative(
    {
      phi: ['m'],
      psi: [
        ['h', 'n'],
        ['a', 'c'],
      ],
      alpha0: 'solve',
    },
    y,
    5,
  );
  for (let i = 1; i < one.length; i++) expect(solved[i]).toBe(one[i]);
  expect(solved[0]).not.toBe(one[0]);
  expect(solved[0] / one[0]).toBeGreaterThan(0);
  expect(solved[0] / one[0]).toBeLessThan(1);
});

test('the slope of c is analytic and stays positive at the -10 mV branch switch', () => {
  expect(model.steadySlope('c', -10)).toBeCloseTo(model.steady(model.rates(-10).c) / 11, 12);
  expect(model.steadySlope('c', -10)).toBeGreaterThan(0);
  expect(model.steadySlope('c', -9.99)).toBe(0);
  for (const v of [-80, -50, -30, -15, -10.5]) {
    const width = 1e-4;
    const difference = (model.steady(model.rates(v + width).c) - model.steady(model.rates(v - width).c)) / (2 * width);
    expect(model.steadySlope('c', v)).toBeCloseTo(difference, 6);
  }
  const weights = model.groupCriteria(['h', 'n', 'c'], -10).weights;
  for (const weight of weights) {
    expect(weight).toBeGreaterThanOrEqual(0);
    expect(weight).toBeLessThanOrEqual(1);
  }
});
