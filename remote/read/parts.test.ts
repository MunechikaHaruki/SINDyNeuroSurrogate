import { expect, test } from 'bun:test';
import { equationsOf, factorOf } from './parts.ts';

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
