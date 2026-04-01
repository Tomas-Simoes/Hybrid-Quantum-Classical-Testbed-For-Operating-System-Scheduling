export PATH := $(HOME)/.local/bin:$(PATH)

.DEFAULT_GOAL := help

.PHONY: help install run add remove freeze activate

help: ## Mostra os comandos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36mmake %-12s\033[0m %s\n", $$1, $$2}'

install: ## Instala todas as dependências  (como npm install)
	uv sync

run: ## Corre o projeto
	uv run python src/main.py

add: ## Adiciona um pacote ao projeto: make add pkg=scipy
	uv add $(pkg)

remove: ## Remove um pacote do projeto: make remove pkg=scipy
	uv remove $(pkg)

freeze: ## Lista os pacotes instalados
	uv pip freeze

activate: ## Mostra o comando para ativar o .venv
	@echo ""
	@echo "  \033[33mCopia e corre este comando:\033[0m"
	@echo ""
	@echo "    source .venv/bin/activate"
	@echo ""

spy:
	uv run py-spy record -o profile.svg -- python src/main.py
	