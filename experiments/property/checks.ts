// 性質の発見。縮約が定常電流の外の入力でも再現するかなど、縮約の性質を確かめる実験の判定。宣言は隣の experiments.json で、主張の名前ごとに判定の関数をここに置く。
import { NAMES, above, all, mean, one, shown, type Judgement, type Row, type Judge } from '../common.ts';

/** 比べる縮約の仕方。 */
const METHODS = ['kam', 'pca', 'ae'];

/** 入力と再現：hn は 3 つの仕方のどれでも Kistler の揺らぐ電流の方が平均 Γ が高く、最も強い揺らぎで線以上なのは KAM だけ。 */
function fluctuatingInput([steady, kistler]: Row[][]): Judgement[] {
  const rises = METHODS.map(m => ({ m, from: mean(one(steady, 'hn', m)), to: mean(one(kistler, 'hn', m)) }));
  const values = kistler[0]?.values ?? [];
  const strongest = values.indexOf(Math.max(...values));
  const atStrongest = METHODS.map(m => ({ m, ok: above(one(kistler, 'hn', m)?.gammas[strongest]) }));
  const passing = atStrongest.filter(a => a.ok === true).map(a => a.m);
  return [
    {
      label: 'hn の平均 Γ（定常電流 → 揺らぐ電流）',
      value: rises.map(r => `${NAMES[r.m]} ${shown(r.from)} → ${shown(r.to)}`).join('・'),
      line: '3 つとも上がる',
      ok: all(rises.map(r => (r.from === null || r.to === null ? null : r.to > r.from))),
    },
    {
      label: `最も強い揺らぎ（${values[strongest] ?? '—'}）で線以上の仕方`,
      value: passing.length > 0 ? passing.map(m => NAMES[m]).join('・') : 'なし',
      line: 'KAM だけ',
      ok: strongest < 0 || atStrongest.some(a => a.ok === null) ? null : passing.length === 1 && passing[0] === 'kam',
    },
  ];
}

/** 主張の名前 → 判定。 */
export const JUDGES: Record<string, Judge> = {
  'fluctuating-input': fluctuatingInput,
};
