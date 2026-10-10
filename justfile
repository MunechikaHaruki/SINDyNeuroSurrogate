## execute with '''just [[command]]'''
## '''just --list''' shows how to use

set shell := ["bash", "-cu"]

PROJECT_NAME := "neurosurrogate"
VIRTUAL_ENV := "uv run"

# port numbering

MLFLOW_PORT := "5100"
MLFLOW_URI := "sqlite:///./mlflow.db"
SMOKE_EXP := "smoke_test"  # just test の main.py run 隔離先 (clean-test が丸ごと削除)

# Delete all compiled Python files
clean-cache:
    find . -type f -name "*.py[co]" -delete
    find . -type d -name "__pycache__" -delete
    find . -type f -name ".DS_Store" -not -path './.git/*' -delete
    rm -rf ./.mypy_cache
    rm -rf ./.ruff_cache
    rm -rf ./.pytest_cache

clean-log:
    rm -rf ./hydra-multiruns
    rm -rf ./hydra-outputs
    rm -rf ./mlruns ./mlflow.db

# Delete all MLflow runs (DB/experiment 構造は保持、artifact ごと物理削除)
clean-run:
    {{ VIRTUAL_ENV }} python -c "import mlflow; mlflow.set_tracking_uri('{{ MLFLOW_URI }}'); c = mlflow.MlflowClient(); [c.delete_run(r.info.run_id) for e in c.search_experiments() for r in c.search_runs(e.experiment_id, run_view_type=1)]"
    {{ VIRTUAL_ENV }} python -m mlflow gc --backend-store-uri {{ MLFLOW_URI }} --tracking-uri {{ MLFLOW_URI }}

# just test が smoke_test experiment に残した run を experiment ごと削除 (本番 run は不変)
clean-test:
    {{ VIRTUAL_ENV }} python -c "import mlflow; mlflow.set_tracking_uri('{{ MLFLOW_URI }}'); c = mlflow.MlflowClient(); e = c.get_experiment_by_name('{{ SMOKE_EXP }}'); [c.delete_run(r.info.run_id) for r in c.search_runs(e.experiment_id, run_view_type=1)] if e else None; c.delete_experiment(e.experiment_id) if e else None"
    {{ VIRTUAL_ENV }} python -m mlflow gc --backend-store-uri {{ MLFLOW_URI }} --tracking-uri {{ MLFLOW_URI }}

# Format source code with ruff (the version pinned by dotfiles, same as the edit hook)
format:
    ruff check --fix
    ruff format

#################################################################################
# static measurement about code                                                 #
#################################################################################

# Lint using ruff
lint:
    ruff format --check
    ruff check
    {{ VIRTUAL_ENV }} mypy .

# Count lines of code
cloc:
    cloc . --vcs=git

# Check code complexity with lizard
lizard:
    {{ VIRTUAL_ENV }} lizard ./neurosurrogate ./scripts

# Check code maintainability with radon
radon:
    {{ VIRTUAL_ENV }} radon cc ./neurosurrogate ./scripts -s -a
    {{ VIRTUAL_ENV }} radon mi ./neurosurrogate ./scripts -s

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

# Smoke test: pytest (ドメイン層) + Hydra entry
test:
    {{ VIRTUAL_ENV }} pytest -q
    MLFLOW_EXPERIMENT={{ SMOKE_EXP }} {{ VIRTUAL_ENV }} python scripts/main.py surrogate=_test_hh_sindy

# traub 系 preset (conf/surrogate/traub_*.yaml) を順に --multirun 実行
traub:
    #!/usr/bin/env bash
    set -euo pipefail
    for f in scripts/conf/surrogate/traub_*.yaml; do
        preset=$(basename "$f" .yaml)
        echo "=== $preset ==="
        {{ VIRTUAL_ENV }} python scripts/main.py --multirun surrogate="$preset"
    done

# activate logging server。Mac ではログイン時の常駐（dotfiles の launchd）が叩くので、手で打たなくてよい。
# 既に上がっていれば何もしない（常駐と手打ちが殺し合わないように）。
mlflow:
    #!/usr/bin/env bash
    set -euo pipefail
    if nc -z 127.0.0.1 {{ MLFLOW_PORT }} 2>/dev/null; then echo "MLflow は既に http://127.0.0.1:{{ MLFLOW_PORT }} で上がっている"; exit 0; fi
    exec {{ VIRTUAL_ENV }} python -m mlflow server --host 127.0.0.1 --port {{ MLFLOW_PORT }} --backend-store-uri {{ MLFLOW_URI }}

# 評価: 系列を原系と学習 run で回し、MLflow の波形 experiment に置く。図は live-textbook の冊が読む
#   just eval traub19_somastim <学習 run の id> ...   (系列の一覧は just eval -h)
eval *args:
    {{ VIRTUAL_ENV }} python scripts/evaluate.py {{ args }}
