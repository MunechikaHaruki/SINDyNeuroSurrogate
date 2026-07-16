## execute with '''just [[command]]'''
## '''just --list''' shows how to use

PROJECT_NAME := "neurosurrogate"
VIRTUAL_ENV := "uv run"

# port numbering

MLFLOW_PORT := "5100"

# Delete all compiled Python files
clean-cache:
    find . -type f -name "*.py[co]" -delete
    find . -type d -name "__pycache__" -delete
    rm -rf ./.mypy_cache
    rm -rf ./.ruff_cache
    rm -rf ./.pytest_cache
    rm -rf ./__marimo__

clean-log:
    rm -rf ./hydra-multiruns
    rm -rf ./hydra-outputs
    rm -rf ./mlruns ./mlflow.db

# Format source code with ruff
format:
    {{ VIRTUAL_ENV }} ruff check --fix
    {{ VIRTUAL_ENV }} ruff format

#################################################################################
# static measurement about code                                                 #
#################################################################################

# Lint using ruff
lint:
    {{ VIRTUAL_ENV }} ruff format --check
    {{ VIRTUAL_ENV }} ruff check
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

# Smoke test: pytest (ドメイン層) + Hydra entry + marimo notebook (cell error -> exit 1)
test:
    {{ VIRTUAL_ENV }} pytest -q
    {{ VIRTUAL_ENV }} python scripts/main.py +surrogate.init.datasets.current.params.duration=180

# Run all surrogate presets (conf/surrogate/*.yaml, excluding _-prefixed)
runall:
    #!/usr/bin/env sh
    set -eu
    presets=$(ls scripts/conf/surrogate/*.yaml | xargs -n1 basename | sed 's/\.yaml$//' | grep -v '^_' | paste -sd, -)
    echo "presets: $presets"
    {{ VIRTUAL_ENV }} python scripts/main.py --multirun surrogate="$presets"

# activate logging server
mlflow:
    @lsof -t -i:{{ MLFLOW_PORT }} | xargs kill -9 || true
    {{ VIRTUAL_ENV }} python -m mlflow ui --port {{ MLFLOW_PORT }} --backend-store-uri sqlite:///./mlflow.db

marimo:
    {{ VIRTUAL_ENV }} marimo edit --watch --no-token --port 2700 scripts/marimo.py

# Claude Code連携用（MCP + watchモード）
marimo-mcp:
    {{ VIRTUAL_ENV }} marimo edit --watch --mcp --no-token --port 2701 scripts/marimo.py
