# One-command entry points. PYTHONPATH is cleared so system site-packages
# (e.g. ROS) cannot shadow the project environment.
PY ?= .venv/bin/python
RUN = PYTHONPATH= $(PY) -m batik

.PHONY: setup data sanity matrix analyse reproduce reproduce-retrain test demo

setup:            ## create .venv with pinned CPU wheels
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
	uv pip install --python $(PY) -r requirements.txt -e .

data:             ## download (~13 GB), audit + cache, leakage-aware split
	$(RUN) download
	$(RUN) audit
	$(RUN) split

sanity:           ## Phase 2 pipeline checks
	$(RUN) sanity

matrix:           ## G1-G4 x 3 seeds (+ G5 supplementary), resumable
	$(RUN) matrix

analyse:          ## tables, figures, robustness, error-analysis evidence, release final model
	$(RUN) aggregate
	$(RUN) robustness
	$(RUN) errors
	$(RUN) release

reproduce:        ## HEADLINE: evaluate released G3 checkpoints on the frozen test split
	$(RUN) verify-split
	$(RUN) reproduce

reproduce-retrain: ## HEADLINE from scratch: retrain G3 x 3 seeds, then evaluate
	$(RUN) reproduce --retrain

test:
	PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PY) -m pytest -q

demo:
	PYTHONPATH= $(PY) demo/app.py
