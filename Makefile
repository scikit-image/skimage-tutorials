.PHONY: clean book

.DEFAULT_GOAL = book

NB_DIR = modules
EXEC_NB_DIR = book/_modules

MARKDOWNS = $(wildcard $(NB_DIR)/*.md)
NOTEBOOKS = $(patsubst $(NB_DIR)/%.md, $(EXEC_NB_DIR)/%.ipynb, $(MARKDOWNS))

_requirements.installed:
	pip install -q -r requirements.txt
	touch _requirements.installed

$(EXEC_NB_DIR):
	mkdir book/_modules

$(EXEC_NB_DIR)/%.ipynb:$(NB_DIR)/%.md $(EXEC_NB_DIR)
	@# Jupytext will also catch and print execution errors
	@# unless a cell is marked with the `raises-exception` tag
	jupytext --execute --to ipynb --output $@ $<

book: _requirements.installed $(NOTEBOOKS)
	@export SPHINXOPTS=-W; make -C book html

clean:
	make -C book clean

