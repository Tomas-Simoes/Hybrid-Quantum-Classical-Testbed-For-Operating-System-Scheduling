export PATH := $(HOME)/.local/bin:$(PATH)

PYTHON ?= python3
UV := $(shell command -v uv 2>/dev/null)
RUN_PYTHON := $(if $(UV),uv run python,$(PYTHON))
RUN_PYSPY := $(if $(UV),uv run py-spy,py-spy)

.DEFAULT_GOAL := help

.PHONY: help install run ui add remove freeze activate test all-tests experiment experiment-all experiment-list spy

help: 		## Mostra os comandos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36mmake %-16s\033[0m %s\n", $$1, $$2}'

install: 	## Instala todas as dependências  (como npm install)
	uv sync

run: 		## Corre o projeto
	$(RUN_PYTHON) src/core/main.py

ui: 		## Corre o projeto em modo UI
	streamlit run src/core/app.py

add: 		## Adiciona um pacote ao projeto: make add pkg=scipy
	uv add $(pkg)

remove: 	## Remove um pacote do projeto: make remove pkg=scipy
	uv remove $(pkg)

freeze: 	## Lista os pacotes instalados
	uv pip freeze

activate: 	## Mostra o comando para ativar o .venv
	@echo ""
	@echo "  \033[33mCopia e corre este comando:\033[0m"
	@echo ""
	@echo "    source .venv/bin/activate"
	@echo ""

all-tests: 	## Corre todos os testes
	$(RUN_PYTHON) -m unittest discover -s tests

test: 		## Corre testes individuais: make test test_brute_force
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		echo "Usage: make test test_brute_force"; \
		echo "       make test test_builder_core"; \
		echo "       make all-tests"; \
		exit 2; \
	fi
	@set -e; \
	for selector in $(filter-out $@,$(MAKECMDGOALS)); do \
		case "$$selector" in \
			brute_force|test_brute_force) module="tests.test_brute_force_solver" ;; \
			builder_core|test_builder|test_builder_core) module="tests.test_builder_core" ;; \
			tests.*) module="$$selector" ;; \
			test_*) module="tests.$$selector" ;; \
			*) module="tests.test_$$selector" ;; \
		esac; \
		echo "Running $$module"; \
		$(RUN_PYTHON) -m unittest "$$module"; \
	done

experiment: 	## Corre cenários por prefixo: make experiment t_1
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		echo "Usage: make experiment t_1"; \
		echo "       make experiment t1_1"; \
		echo "       make experiment-list"; \
		exit 2; \
	fi
	$(RUN_PYTHON) src/core/experiments/scenario_runner.py $(filter-out $@,$(MAKECMDGOALS))

experiment-all: ## Corre todos os cenários TOML
	$(RUN_PYTHON) src/core/experiments/scenario_runner.py --all

experiment-list: ## Lista os cenários TOML disponíveis
	$(RUN_PYTHON) src/core/experiments/scenario_runner.py --list

spy: 		## Grava um perfil py-spy em profile.svg
	$(RUN_PYSPY) record -o profile.svg -- $(PYTHON) src/core/main.py

# Allows commands like `make experiment t_1` and `make test test_brute_force` by swallowing extra selector goals.
%:
	@:
