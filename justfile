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
	rm -rf ./mlruns

clean:
	rm -rf ~/.prefect


#Format source code with ruff
format:
	{{VIRTUAL_ENV}} ruff check --fix
	{{VIRTUAL_ENV}} ruff format

#################################################################################
# static measurement about code                                                 #
#################################################################################

#Lint using ruff
lint:
	{{VIRTUAL_ENV}} ruff format --check
	{{VIRTUAL_ENV}} ruff check
	{{VIRTUAL_ENV}} mypy .
	{{VIRTUAL_ENV}} pylint neurosurrogate scripts

#Count lines of code
cloc:
	cloc . --vcs=git

#Check code complexity with lizard
lizard:
	{{VIRTUAL_ENV}} lizard ./neurosurrogate ./scripts ./data_scripts --exclude "build/*"

#Check code maintainability with radon
radon:
	{{VIRTUAL_ENV}} radon cc ./neurosurrogate ./scripts -s -a
	{{VIRTUAL_ENV}} radon mi ./neurosurrogate ./scripts -s

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################
# build
build:
	{{VIRTUAL_ENV}} python setup.py build_ext --inplace
rebuild:
	rm -rf ./build
	{{VIRTUAL_ENV}} python setup.py build_ext --inplace

# activate prefect server
prefect:
	{{VIRTUAL_ENV}} prefect server start

# activate logging server
mlflow:
    @lsof -t -i:{{MLFLOW_PORT}} | xargs kill -9 || true
    {{VIRTUAL_ENV}} python -m mlflow ui --port {{MLFLOW_PORT}} --backend-store-uri ./mlruns

# deactivate で仮想環境から抜けれます
alias:
	source .venv/bin/activate
	eval "$(python scripts/main.py -sc install=zsh)"
	alias p='python scripts/main.py'