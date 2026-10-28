# One-command entry points. PYTHONPATH is cleared so system site-packages
# (e.g. ROS) cannot shadow the project environment.
PY ?= .venv/bin/python
RUN = PYTHONPATH= $(PY) -m batik

# Released checkpoints (24 x best.pt + final model), GitHub Release v1.0
CKPT_URL ?= https://github.com/Universitas-Cakrawala/semarang-batik-fine-grained-transfer-learning/releases/download/v1.0/batik_checkpoints_v1.0.tar.gz
CKPT_SHA256 = c4898ab993d7e877e2b9ebaa403c781161dbb57402ed12f8489ec9b90e9d7350
CKPT_TGZ = artifacts/batik_checkpoints_v1.0.tar.gz

.PHONY: setup data checkpoints sanity matrix analyse reproduce reproduce-retrain test demo

setup:            ## create .venv with pinned CPU wheels
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
	uv pip install --python $(PY) -r requirements.txt -e .

data:             ## download (~13 GB), audit + cache, verify the frozen split (new split: batik split --force)
	$(RUN) download
	$(RUN) audit
	$(RUN) verify-split

checkpoints:      ## download + verify released checkpoints into artifacts/runs/*/best.pt and artifacts/release/
	curl -fL --retry 5 -o $(CKPT_TGZ) $(CKPT_URL)
	$(PY) -c "import hashlib,sys; d=hashlib.sha256(open('$(CKPT_TGZ)','rb').read()).hexdigest(); sys.exit(0 if d=='$(CKPT_SHA256)' else 'checkpoint archive checksum mismatch: '+d)"
	tar xzf $(CKPT_TGZ) -C artifacts
	rm $(CKPT_TGZ)

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
