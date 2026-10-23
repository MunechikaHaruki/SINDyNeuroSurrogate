<!--
  区画のつながり（Observable Plot）。区画を点、つながりを線で描き、線の中ほどに重みを書く。

    <NetChart nodes={[{ name: 'soma', x: 0, y: 0, group: '学習した区画', stim: true }, …]}
      edges={[{ from: 'soma', to: 'c01', label: '0.5' }]} />

  位置 x（列）と y（段）は図が決めて渡し、ここは並べて描くだけ。点の色は group ごとに凡例に並び、
  stim の区画は輪で囲む。幅は置いた場所に合わせ、列が多ければ横に流れる。
-->
<script lang="ts">
  import * as Plot from '@observablehq/plot';
  import { PALETTE } from '$lib/palette.ts';

  type Node = { name: string; x: number; y: number; group: string; stim: boolean };
  let {
    nodes,
    edges,
  }: {
    nodes: Node[];
    edges: { from: string; to: string; label: string }[];
  } = $props();
  let container = $state<HTMLDivElement>();
  let width = $state(640);

  const GROUP_COLORS = [PALETTE.green, PALETTE.blue, PALETTE.neutral, PALETTE.amber];
  const COLUMN = 56;
  const ROW = 64;

  $effect(() => {
    if (!container) return;
    const at = new Map(nodes.map(n => [n.name, n]));
    const links = edges.filter(e => at.has(e.from) && at.has(e.to)).map(e => ({ ...e, a: at.get(e.from)!, b: at.get(e.to)! }));
    const columns = Math.max(1, ...nodes.map(n => n.x + 1));
    const rows = Math.max(1, ...nodes.map(n => n.y + 1));
    const groups = [...new Set(nodes.map(n => n.group))];
    const chart = Plot.plot({
      width: Math.max(width || 640, columns * COLUMN + 40),
      height: rows * ROW + 70,
      margin: 30,
      ariaLabel: '区画のつながり',
      style: { background: 'transparent', fontFamily: 'inherit', fontSize: '12px' },
      x: { domain: [-0.5, columns - 0.5], axis: null },
      y: { domain: [rows - 0.5, -0.5], axis: null },
      color: { domain: groups, range: GROUP_COLORS, legend: true },
      marks: [
        Plot.link(links, { x1: l => l.a.x, y1: l => l.a.y, x2: l => l.b.x, y2: l => l.b.y, stroke: PALETTE.neutral, strokeWidth: 1.5 }),
        Plot.text(links, { x: l => (l.a.x + l.b.x) / 2, y: l => (l.a.y + l.b.y) / 2, text: 'label', dy: -8, fill: PALETTE.subtle, fontSize: 10 }),
        Plot.dot(nodes, { x: 'x', y: 'y', r: 11, fill: 'group', stroke: 'white' }),
        Plot.dot(
          nodes.filter(n => n.stim),
          { x: 'x', y: 'y', r: 15, stroke: PALETTE.rust, strokeWidth: 3 },
        ),
        Plot.text(nodes, { x: 'x', y: 'y', text: 'name', dy: 22, fill: PALETTE.ink }),
      ],
    });
    container.replaceChildren(chart);
    return () => chart.remove();
  });
</script>

<div class="overflow-x-auto" bind:this={container} bind:clientWidth={width}></div>
