## execute with '''just [[command]]'''
## '''just --list''' shows how to use

set shell := ["bash", "-cu"]

PROJECT_NAME := "neurosurrogate"
VIRTUAL_ENV := "uv run"

ORCHESTRATE := "bun evaluation/orchestrate/orchestrate.ts"

# Delete all compiled Python files
clean-cache:
    find . -type f -name "*.py[co]" -delete
    find . -type d -name "__pycache__" -delete
    find . -type f -name ".DS_Store" -not -path './.git/*' -delete
    rm -rf ./.mypy_cache
    rm -rf ./.ruff_cache
    rm -rf ./.pytest_cache

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
    {{ VIRTUAL_ENV }} lizard ./neurosurrogate

# Check code maintainability with radon
radon:
    {{ VIRTUAL_ENV }} radon cc ./neurosurrogate -s -a
    {{ VIRTUAL_ENV }} radon mi ./neurosurrogate -s

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

# Smoke test: pytest (ドメイン層と neurosurrogate/run.py) + 段取りから学習 1 本 (置き場は使い捨て)
test:
    #!/usr/bin/env bash
    set -euo pipefail
    {{ VIRTUAL_ENV }} pytest -q
    runs=$(mktemp -d)
    trap 'rm -rf "$runs"' EXIT
    {{ ORCHESTRATE }} train _test_hh_sindy --runs "$runs"
