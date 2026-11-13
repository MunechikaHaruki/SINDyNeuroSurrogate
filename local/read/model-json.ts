// 学習 run 1 本の中身（description/model.json。形は ../shared/runs.ts）を表示するための変換。手元の画面だけが使う。
// 指標や再構成誤差の計算は計算する機械が済ませて置いている。
import type { NetJson } from '../../shared/runs.ts';

/** 式の表示。sympy は下付きの「m hh」を空白でつなぐので、model 名を括弧に入れて読めるようにする（α_{m(hh)}）。 */
export function targetTex(latex: string): string {
  return latex.replace(/_\{(\w+) (\w+)\}/g, '_{$1($2)}');
}

/** 特徴量の表示。1 つの表の項はどれも同じ model 由来なので、下付きの model 名を落とす（α_{m}）。 */
export function featureTex(latex: string): string {
  return latex.replace(/_\{(\w+) \w+\}/g, '_{$1}');
}

export type Placed = { name: string; x: number; y: number };

/**
 * ネットの配置。刺激の区画から最も遠い区画を根に取り、根からの辺の数を列にする。
 * 鎖は一列に並び（刺激が中央にあっても片側へ伸びる）、枝分かれは同じ列に縦に並ぶ。
 */
export function placeNet(net: NetJson): Placed[] {
  const fromStim = distances(net, net.stim);
  let root = net.stim;
  for (const node of net.nodes) if ((fromStim[node.name] ?? -1) > (fromStim[root] ?? -1)) root = node.name;
  const depth = distances(net, root);
  const rows: Record<number, number> = {};
  const placed: Placed[] = [];
  for (const node of net.nodes) {
    const column = depth[node.name] ?? 0;
    const row = rows[column] ?? 0;
    rows[column] = row + 1;
    placed.push({ name: node.name, x: column, y: row });
  }
  return placed;
}

/** 区画 `start` から各区画への辺の数（向きは見ない）。届かない区画は入らない。 */
function distances(net: NetJson, start: string): Record<string, number> {
  const neighbors: Record<string, string[]> = {};
  for (const node of net.nodes) neighbors[node.name] = [];
  for (const edge of net.edges) {
    neighbors[edge.src].push(edge.dst);
    neighbors[edge.dst].push(edge.src);
  }
  const found: Record<string, number> = { [start]: 0 };
  const queue = [start];
  for (let head = 0; head < queue.length; head++) {
    const here = queue[head];
    for (const next of neighbors[here] ?? []) {
      if (next in found) continue;
      found[next] = found[here] + 1;
      queue.push(next);
    }
  }
  return found;
}

/** 累積寄与率。i 番目は先頭から i 番目の成分までの寄与率の和。 */
export function cumulative(ratios: number[]): number[] {
  const list: number[] = [];
  let total = 0;
  for (const ratio of ratios) {
    total += ratio;
    list.push(total);
  }
  return list;
}
