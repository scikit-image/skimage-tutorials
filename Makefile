OBSHELL=/bin/bash

.DEFAULT_GOAL = html

LESSONS_DIR = lessons
GENERATED_LESSONS_DIR = book/lessons

_requirements.installed:
	pip install -q -r requirements.txt
	touch _requirements.installed

MARKDOWNS = $(wildcard $(LESSONS_DIR)/*.md)
MD_OUTPUTS = $(patsubst $(LESSONS_DIR)/%.md, $(GENERATED_LESSONS_DIR)/%.md, $(MARKDOWNS))
NOTEBOOKS = $(patsubst %.md, %.ipynb, $(MD_OUTPUTS))

.SECONDARY: $(MD_OUTPUTS) $(NOTEBOOKS)

$(GENERATED_LESSONS_DIR)/%.ipynb:$(LESSONS_DIR)/%.md book/lessons book/lessons/images
        # This does not work, due to bug in notedown; see https://github.com/aaren/notedown/issues/53
	#notedown --match=python --precode='%matplotlib inline' $< > $@
	notedown --match=python $< > $@
	jupyter nbconvert --execute --inplace $@ --ExecutePreprocessor.timeout=-1

%.md:%.ipynb
	jupyter nbconvert --to=mdoutput --output="$(notdir $@)" --output-dir=$(GENERATED_LESSONS_DIR) $<
#	$(eval NBSTRING := [📂 Download lesson notebook](.\/$(basename $(notdir $@)).ipynb)\n\n)
#	sed -i'.bak' '1s/^/$(NBSTRING)/' $@


book/lessons:
	mkdir -p book/lessons

book/lessons/images:
	ln -s ${PWD}/lessons/images ${PWD}/book/lessons/images

html: | _requirements.installed $(NOTEBOOKS) $(MD_OUTPUTS)
	@export SPHINXOPTS=-W; make -C book html
	cp $(GENERATED_LESSONS_DIR)/*.ipynb book/build/html/lessons/

clean:
	rm -rf $(GENERATED_LESSONS_DIR)/*
