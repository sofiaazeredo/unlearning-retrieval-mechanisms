PY ?= python

smoke:            ## 1-minute dry run on a toy model (no downloads)
	$(PY) -m pytest -q tests

data:             ## download CounterFact
	bash data/download_counterfact.sh

main:             ## the full GPT-2 study (about 1-2 hours on a T4)
	$(PY) run_experiments.py --config configs/main.json

analyze:          ## rebuild tables and figures from saved results
	$(PY) run_experiments.py --config configs/main.json --stages analyze

.PHONY: smoke data main analyze
