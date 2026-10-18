# SINDyNeuroSurrogate

[**SINDy (Sparse Identification of Nonlinear Dynamics)**](https://arxiv.org/abs/1904.02107)を用いて、神経細胞モデル（Hodgkin-Huxleyモデル等）の代理モデルを構築し、シミュレーションの高速化を目指す研究プロジェクト


脳の大規模シミュレーションにおいて、詳細な神経細胞モデルは計算コストが非常に高いという課題があります。
本プロジェクトでは、機械学習手法の一つである [SINDy](https://arxiv.org/abs/1904.02107) を用いて、複雑な非線形ダイナミクスをスパースな方程式として同定し、計算コストの低い代理モデル（Surrogate Model）への置き換えを試みています。

# Requirements
* uv
* just

## setup
```bash
uv sync
just train traub_useMC19CompForTeachingData          # 学習
just eval traub19_somastim poster_traub_pca          # 評価（preset か学習 run のハッシュ）
```

何を回すかは `research/50-59-research/53-evaluation/conf.ts` に書き、結果は `runs/<spec のハッシュ>/` に入る。同じ条件は回し直さない。`research/50-59-research/53-evaluation/` の図（`53.03-trained-model` が学習 run、`53.04-evaluation` が評価）が読んで描く。図は context-router の開発サーバー（`http://127.0.0.1:4321/`）で開く。

## Usage
![Status](https://img.shields.io/badge/docs-writing-orange)
