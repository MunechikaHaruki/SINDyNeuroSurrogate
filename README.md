# SINDyNeuroSurrogate

[**SINDy (Sparse Identification of Nonlinear Dynamics)**](https://arxiv.org/abs/1904.02107)を用いて、神経細胞モデル（Hodgkin-Huxleyモデル等）の代理モデルを構築し、シミュレーションの高速化を目指す研究プロジェクト


脳の大規模シミュレーションにおいて、詳細な神経細胞モデルは計算コストが非常に高いという課題があります。
本プロジェクトでは、機械学習手法の一つである [SINDy](https://arxiv.org/abs/1904.02107) を用いて、複雑な非線形ダイナミクスをスパースな方程式として同定し、計算コストの低い代理モデル（Surrogate Model）への置き換えを試みています。

# Requirements
* uv
* bun
* just

## setup
```bash
uv sync
```

何を回すかは `evaluation/orchestrate/conf.ts` に書き、学習と評価は操作盤の冊 `evaluation/53.05-runs` のボタンから回す。結果は `runs/<spec のハッシュ>/` に入り、同じ条件は回し直さない。`evaluation/` の図（`53.03-trained-model` が学習 run、`53.04-evaluation` が評価）が読んで描く。冊は context-router の開発サーバー（`http://127.0.0.1:4321/`）で開く。

## Usage
![Status](https://img.shields.io/badge/docs-writing-orange)
