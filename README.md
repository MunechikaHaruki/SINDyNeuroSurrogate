# SINDyNeuroSurrogate

[**SINDy (Sparse Identification of Nonlinear Dynamics)**](https://arxiv.org/abs/1904.02107) を用いて、Hodgkin-Huxley ニューロンモデルのサロゲートモデルを構築し、大規模脳シミュレーションの高速化を目指す研究プロジェクトです。

## 背景と目的

脳の大規模シミュレーションでは、詳細な神経細胞モデル（HH モデル）の計算コストが問題になります。
本プロジェクトでは SINDy でスパースな支配方程式を同定し、軽量なサロゲートモデルへの置き換えを試みています。

**アプローチ：**

1. HH モデルでシミュレーションして訓練データを生成
2. PCA でゲート変数 (m, h, n) を 1 次元潜在変数に圧縮
3. SINDy でスパース微分方程式を発見
4. サロゲートをマルチコンパートメントネットワークに組み込んで評価

## Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (パッケージ管理)
- [just](https://just.systems/) (タスクランナー)

主な依存ライブラリは `pyproject.toml` を参照してください。

```toml
pysindy ~= 2.0.0
hydra-core >= 1.3.2
mlflow >= 3.3.2
numba >= 0.63.1
xarray >= 2024.7.0
scikit-learn < 1.6.0
```

## セットアップ

```bash
uv sync
```

## 使い方

```bash
just --list   # 利用可能なコマンドを確認
```

### 学習・評価

```bash
# サロゲートモデルを学習して評価
uv run python scripts/main.py

# パラメータスイープ（例：正則化強度を変えて比較）
uv run python scripts/main.py --multirun sindy.optimizer.alpha=0.01,0.1,1.0,2.0
```

### 実験ログの可視化

```bash
just mlflow    # MLflow UI を起動 (port 5100)
```

### 方程式の確認

```bash
just marimo    # marimo ノートブックで HH 方程式を確認
```

### コードの整形・検査

```bash
just format    # ruff で自動整形
just lint      # ruff + mypy + pylint
```

### クリーンアップ

```bash
just clean-cache   # __pycache__ などを削除
just clean-log     # hydra / mlflow のログを削除
```

## プロジェクト構成

```
neurosurrogate/
├── modeling/
│   ├── neuron_core.py     # HH モデルの数式 (Numba JIT)
│   ├── calc_engine.py     # 統合シミュレータ
│   ├── __init__.py        # SINDySurrogateWrapper, PCAPreProcessorWrapper
│   └── profiler.py        # 精度・計算コスト分析
└── utils/
    ├── current_generators.py  # 入力電流パターン生成
    └── plots.py               # 可視化

scripts/
├── main.py    # 学習パイプライン
├── eval.py    # モデル評価
└── utils/     # Builder 群・Hydra 設定
```
