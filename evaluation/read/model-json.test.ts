import { describe, expect, test } from 'bun:test';
import { cumulative, featureTex, histogram, placeNet, targetTex, type NetJson } from './model-json.ts';

describe('式の表示', () => {
  // sympy の sp.latex の出力そのまま（この repo の smoke run の model.json から）。
  const rate = String.raw`z_{1} \alpha_{m hh}{\left(V \right)}`;

  test('特徴量は下付きの model 名を落とす', () => {
    expect(featureTex(rate)).toBe(String.raw`z_{1} \alpha_{m}{\left(V \right)}`);
  });

  test('式は model 名を括弧に入れて残す', () => {
    expect(targetTex(rate)).toBe(String.raw`z_{1} \alpha_{m(hh)}{\left(V \right)}`);
  });

  test('下付きに空白の無い記号は変えない', () => {
    expect(featureTex('z_{1}')).toBe('z_{1}');
    expect(targetTex('V')).toBe('V');
  });
});

describe('ネットの配置', () => {
  function chain(names: string[], stim: string): NetJson {
    const edges: NetJson['edges'] = [];
    for (let i = 1; i < names.length; i++) edges.push({ src: names[i - 1], dst: names[i], weight: 1 });
    const nodes: NetJson['nodes'] = [];
    for (const name of names) nodes.push({ name, type: 'hh' });
    return { nodes, edges, stim, replaceable: [], train: [] };
  }

  test('刺激が鎖の中央にあっても、一列に並ぶ', () => {
    const placed = placeNet(chain(['a', 'b', 'soma', 'c', 'd'], 'soma'));
    const columns: number[] = [];
    for (const node of placed) {
      columns.push(node.x);
      expect(node.y).toBe(0);
    }
    expect(columns.sort()).toEqual([0, 1, 2, 3, 4]);
  });

  test('枝分かれは同じ列に縦に並ぶ', () => {
    const net = chain(['root', 'soma'], 'soma');
    net.nodes.push({ name: 'x', type: 'hh' }, { name: 'y', type: 'hh' });
    net.edges.push({ src: 'soma', dst: 'x', weight: 1 }, { src: 'soma', dst: 'y', weight: 1 });
    const placed = placeNet(net);
    const x = placed.find(node => node.name === 'x')!;
    const y = placed.find(node => node.name === 'y')!;
    expect(x.x).toBe(y.x);
    expect(x.y).not.toBe(y.y);
  });

  test('区画が 1 つなら原点に置く', () => {
    expect(placeNet(chain(['soma'], 'soma'))).toEqual([{ name: 'soma', x: 0, y: 0 }]);
  });
});

describe('表示のための変換', () => {
  test('ヒストグラムは範囲の中の有限な値だけを数え、上端は最後の区間に入れる', () => {
    const bins = histogram(new Float64Array([0, 0.5, 1, 2, NaN, -1]), 0, 2, 2);
    expect(bins).toEqual([
      { center: 0.5, count: 2 },
      { center: 1.5, count: 2 },
    ]);
  });

  test('累積寄与率は先頭からの和', () => {
    const sums = cumulative([0.5, 0.25, 0.125]);
    expect(sums).toEqual([0.5, 0.75, 0.875]);
  });
});
