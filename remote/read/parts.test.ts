import { expect, test } from 'bun:test';
import { equationsOf, factorOf, levelsOf, stableRoot, steadyOf, strided } from './parts.ts';
import { sameScale } from './inputs.ts';

// sympy の sp.latex の出力そのまま（研究コードの項のカタログの形）。
const ALPHA = String.raw`\alpha_{n traub}{\left(V \right)}`;
const BETA = String.raw`\beta_{n traub}{\left(V \right)}`;

test('a term splits into its rate factor and the rest, which is 1 when nothing is left', () => {
  expect(factorOf(ALPHA)).toEqual({ rate: ALPHA, monomial: '1' });
  expect(factorOf(`z_{1} ${ALPHA}`)).toEqual({ rate: ALPHA, monomial: 'z_{1}' });
  const sum = String.raw`\left(${ALPHA} + ${BETA}\right)`;
  expect(factorOf(`z_{1} ${sum}`)).toEqual({ rate: sum, monomial: 'z_{1}' });
  expect(factorOf('V z_{1}^{2}')).toEqual({ rate: '', monomial: 'V z_{1}^{2}' });
  expect(factorOf('1')).toEqual({ rate: '', monomial: '1' });
});

test('equations group the terms by rate factor in library order, terms without a rate last', () => {
  const [equation] = equationsOf({
    targets: ['z_{1}'],
    features: [ALPHA, BETA, `z_{1} ${ALPHA}`, 'V', '1'],
    xi: [[1, 0, -1, 0.5, 2]],
  });
  expect(equation.target).toBe('z_{1}');
  expect(equation.groups).toEqual([
    {
      rate: ALPHA,
      terms: [
        { monomial: '1', xi: 1 },
        { monomial: 'z_{1}', xi: -1 },
      ],
    },
    { rate: BETA, terms: [{ monomial: '1', xi: 0 }] },
    {
      rate: '',
      terms: [
        { monomial: 'V', xi: 0.5 },
        { monomial: '1', xi: 2 },
      ],
    },
  ]);
});

test('the gate-space trail keeps every step-th point within the range, up to the limit, and drops non-finite points', () => {
  const h = new Float64Array([0, 0.1, 0.2, 0.3, NaN, 0.5]);
  const n = new Float64Array([1, 0.9, 0.8, 0.7, 0.6, 0.5]);
  expect(strided([h, n], 1, 5, 3)).toEqual([
    [0.1, 0.9],
    [0.3, 0.7],
    [0.5, 0.5],
  ]);
  expect(strided([h, n], 0, 3, 10)).toHaveLength(4);
});

test('the encoder grid names its gates by position in the group, and a learner without one (PCA) has no levels', () => {
  const plane = {
    planes: [
      {
        x: 'N',
        y: 'H',
        xs: [0, 1],
        ys: [0, 1],
        latents: [
          {
            latent: 'z1',
            values: [
              [0, 1],
              [1, null],
            ],
          },
        ],
      },
    ],
  };
  expect(levelsOf(plane, ['H', 'N'])).toEqual([
    {
      x: 1,
      y: 0,
      xs: [0, 1],
      ys: [0, 1],
      latents: [
        {
          latent: 'z1',
          values: [
            [0, 1],
            [1, null],
          ],
        },
      ],
    },
  ]);
  expect(levelsOf(null, ['H', 'N'])).toEqual([]);
});

test('a phase plane has the same scale when both axes share a unit: two latents, or V with a KAM equivalent potential', () => {
  expect(sameScale({ x: 'h+n.z1', y: 'a+c.z1' }, false)).toBe(true);
  expect(sameScale({ x: 'V', y: 'h+n.z1' }, false)).toBe(false);
  expect(sameScale({ x: 'V', y: 'h+n.z1' }, true)).toBe(true);
});

test('the steady latent is where the right-hand side falls through zero as z grows, and tau is minus one over its slope there', () => {
  // f = 2 − z（z∞ = 2、∂f/∂z = −1 なので τ = 1）
  const z = [0, 1, 2, 3, 4];
  expect(stableRoot(z, [2, 1, 0, -1, -2], null)).toEqual({ z: 2, slope: -1 });
  // 負から正へ横切る 0 は不安定なので取らない
  expect(stableRoot(z, [-2, -1, 0.5, 1, 2], null)).toBeNull();
  // 安定な 0 が 2 つなら、前の列に近い方
  expect(stableRoot(z, [1, -1, 1, -1, -1], 2.4)?.z).toBe(2.5);
  const grid = [
    [2, 4],
    [1, 2],
    [0, 0],
    [-1, -2],
    [-2, -4],
  ];
  const { steady, tau } = steadyOf([-60, -40], z, grid, '学習器');
  expect(steady.points).toEqual([
    { x: -60, y: 2 },
    { x: -40, y: 2 },
  ]);
  expect(tau.points).toEqual([
    { x: -60, y: 1 },
    { x: -40, y: 0.5 },
  ]);
});
