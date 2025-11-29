# SINDyNeuroSurrogate

[**SINDy (Sparse Identification of Nonlinear Dynamics)**](https://arxiv.org/abs/1711.05501)を用いて、神経細胞モデル（Hodgkin-Huxleyモデル等）の代理モデルを構築し、シミュレーションの高速化を目指す研究プロジェクトです。

## 🔬 背景と目的 (Background)
脳の大規模シミュレーションにおいて、詳細な神経細胞モデルは計算コストが非常に高いという課題があります。
本プロジェクトでは、機械学習手法の一つである [SINDy](https://arxiv.org/abs/1711.05501) を用いて、複雑な非線形ダイナミクスをスパースな方程式として同定し、計算コストの低い代理モデル（Surrogate Model）への置き換えを試みています。

## 🛠️ 技術スタック (Tech Stack)
* **Language**: Python
* **ML/Math**: PySINDy, NumPy, SciPy, Scikit-learn
* **Pipeline**: Gokart, Luigi
* **Config**: Hydra


## Usage
![Status](https://img.shields.io/badge/docs-writing-orange)

鋭意研究中