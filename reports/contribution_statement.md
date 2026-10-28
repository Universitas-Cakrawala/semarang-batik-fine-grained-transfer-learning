# Individual Contribution Statement — Kelompok 5 (Track A)

Each member lists what they did, with links to evidence (files, commits, logs).
The default ownership agreed in planning is pre-filled. **Edit it to match
what actually happened.** Each member also confirms that they can explain their
code in the individual viva.

| Member | Ownership | Evidence in repository | Code reviewed for (≥1 other member) | Reproduced end-to-end (yes/no + date) | Signature |
|---|---|---|---|---|---|
| Tita Noviana | Dataset audit, data card, split, leakage check | `src/batik/data/{audit,split}.py`, `data/manifests/`, `DATA_LICENSE.md` | Fajar (`Makefile`, `README.md`): found `PY` hard-coded to `.venv/bin/python`, which doesn't match a native-Windows venv layout, so `make` targets don't run outside WSL/Git Bash/Colab. Suggested a README note on 2026-10-28. | Yes, 2026-10-28 (Google Colab, CPU). Ran `make setup`, `make data`, `make checkpoints`, `make reproduce` (evaluates released G3 checkpoints, no retraining). Result: `HEADLINE G3 EfficientNet-B0 test Macro F1 = 100.00 ± 0.00 %` over seeds 42/43/44, matching README line 52. First attempt stalled at `make data` (Colab's inline `MPLBACKEND` isn't accepted by the project venv); fixed with `MPLBACKEND=Agg` and resumed. | |
| Titanio Yudista | Model architectures, training loop | `src/batik/{models,train}.py`, `configs/`, `artifacts/runs/*/history.csv` | | | |
| Suci Fransisca Sisilia R. | Evaluation, ablation, robustness, error analysis | `src/batik/{analysis,robustness,errors,diagnostics}.py`, `artifacts/metrics/`, `figures/errors/` | | | |
| Fajar Dwiharjo | Reproducibility, repo integration, demo | `README.md`, `Makefile`, `requirements.txt`, `notebooks/colab_runner.ipynb`, `demo/app.py` | | | |

Ownership changes (date, what moved, why):

-

We confirm the statement above is accurate.

Tita Noviana: Tita Noviana  Titanio Yudista: ____________ Suci Fransisca Sisilia R.: ____________ Fajar Dwiharjo: ____________
