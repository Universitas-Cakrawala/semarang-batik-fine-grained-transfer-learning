# Semarang Batik: From-Scratch CNN vs Transfer Learning

UAS Deep Learning, **Kelompok 5, Track A** (Vision classification with transfer learning).
Members: Tita Noviana, Titanio Yudista, Suci Fransisca Sisilia R., Fajar Dwiharjo.

This project compares a from-scratch Custom CNN, EfficientNet-B0 trained from
random initialisation, and EfficientNet-B0 initialised from ImageNet on 10 Semarang
batik motifs. It also includes:

- an augmentation ablation;
- a supplementary two-stage fine-tuning run;
- a *shortcut-controlled* grayscale protocol;
- an evaluation-only robustness benchmark;
- error analysis with causal interventions.

Deviations from the approved proposal are listed in
[Deviations from the proposal](#deviations-from-the-proposal).

> **Headline number:** see [Results](#results). Reproduce it with
> **`make setup && make data && make checkpoints && make reproduce`** (see [Reproduce](#reproduce)).

## Key findings about the data (read before the results)

| Finding | Evidence | Consequence |
|---|---|---|
| The Kaggle snapshot stores each capture as ~4 rotated/flipped copies | Same-second captures have rotation-invariant pHash distance 0–4. Ordinary pHash gives ~30 | A plain random split (as in the proposal) puts a rotated copy of **98.9%** of test images into train. We split by **source photo** groups instead |
| Each class is one cloth with its own colourway | HSV colour histogram + logistic regression: **100%** test Macro F1 | RGB results hit the ceiling. We added a grayscale + histogram-equalisation protocol (intensity-histogram shortcut falls to 55.7%, LBP texture stays 96.3%) |
| All photos per class come from one session and one cloth | EXIF timestamps (5–18 min per class) | Results measure generalisation to new views of the *same* cloth, not to other cloths with the same motif |

## Setup

```bash
make setup              # uv venv + pinned CPU wheels (see requirements.txt for CUDA)
# or: python -m venv .venv && pip install torch==2.5.1 torchvision==0.20.1 && pip install -r requirements.txt -e .
```

Python 3.12, PyTorch 2.5.1, and torchvision 0.20.1. On machines where a system
`PYTHONPATH` (for example ROS) shadows packages, the Makefile clears it.

## Reproduce

```bash
make data               # download Kaggle v2 (~13 GB, no credentials needed), audit + cache, verify split
make checkpoints        # download + SHA-256-verify the released checkpoints (GitHub Release v1.0, 283 MB)
make reproduce          # HEADLINE: evaluate the released G3 checkpoints on the frozen test split
make reproduce-retrain  # same headline, retraining G3 x 3 seeds from scratch
```

`make reproduce` checks the SHA-256 of `data/manifests/split.csv` and refuses to
report a number if the split differs. `make checkpoints` places the checkpoints in
`artifacts/runs/<run_id>/best.pt` (see [Saved model](#saved-model)).
Expected output: `HEADLINE  G3 EfficientNet-B0 (ImageNet, aug ON) test Macro F1 = 100.00 ± 0.00 %`.

Full pipeline, step by step (each step writes auditable artefacts):

| Step | Command | Output |
|---|---|---|
| Download + checksum | `python -m batik download` | `data/raw/`, `data/manifests/dataset_snapshot.json` |
| Audit + cache + EDA | `python -m batik audit` | `inventory.csv`, `exclusions.csv`, `audit_summary.json`, `figures/` |
| Leakage-aware split | `python -m batik split` (refuses to overwrite) | `split.csv`, `split_meta.json`, `near_duplicate_edges.csv`, `cross_split_nearest_train.csv` |
| Verify frozen split | `python -m batik verify-split` (run by `make data` and `make reproduce`) | checksum OK / refuses |
| Dataset diagnostics | `python -m batik diagnostics` | `artifacts/metrics/dataset_diagnostics.json` |
| Sanity checks | `python -m batik sanity` | `artifacts/sanity/sanity_report.json`, `figures/augmentation_grid.png` |
| Experiment matrix | `python -m batik matrix` (resumable) | `artifacts/runs/<run_id>/` |
| Tables + figures | `python -m batik aggregate` | `artifacts/metrics/{tables.md,summary.json,final_model.json}` |
| Robustness | `python -m batik robustness` | `artifacts/metrics/robustness.{csv,md}` |
| Error analysis | `python -m batik errors` | `artifacts/error_analysis/<run_id>/`, `figures/errors/<run_id>/` |
| Release final model | `python -m batik release` | `artifacts/release/batik_effb0_final.pt`, `release.json` |
| All analysis steps above | `make analyse` | aggregate → robustness → errors → release |
| GPU (Colab) | `notebooks/colab_runner.ipynb` (clones `main`, runs and stores runs on Google Drive) | same, on a T4 |

## Experiment matrix

| Group | Model | Init | Input | Aug | Seeds | Role |
|---|---|---|---|---|---|---|
| G1 | Custom 4-block CNN (0.39 M params) | random | RGB | ON | 42/43/44 | from-scratch baseline |
| G2 | EfficientNet-B0 (4.0 M) | random | RGB | ON | 42/43/44 | architecture control |
| G3 | EfficientNet-B0 | ImageNet | RGB | ON | 42/43/44 | **headline transfer model** |
| G4 | EfficientNet-B0 | ImageNet | RGB | OFF | 42/43/44 | augmentation ablation (vs G3) |
| G1g–G3g | as G1–G3 | as G1–G3 | grayscale + equalised | ON | 42/43/44 | shortcut-controlled protocol |
| G5 | EfficientNet-B0, linear probe → unfreeze top blocks | ImageNet | RGB | ON | 42/43/44 | supplementary training strategy |

Shared protocol for all runs:
- AdamW, lr 3e-4, weight decay 1e-4, batch size 32, 30 epochs, per-step cosine
  schedule.
- Checkpoint chosen by validation Macro F1; test set evaluated once.
- Split is group-stratified 70/15/15 with seed 42 and is frozen.

## Results

Test Macro F1 on the frozen 450-image test split, mean ± std over seeds 42/43/44
(`artifacts/metrics/tables.md`). Trained on a Colab Tesla T4.

| Group | Model | Input | Test Macro F1 (%) |
|---|---|---|---:|
| G1 | Custom CNN, scratch (baseline) | RGB | 100.00 ± 0.00 |
| G2 | EfficientNet-B0, scratch | RGB | 100.00 ± 0.00 |
| **G3** | **EfficientNet-B0, ImageNet (headline)** | RGB | **100.00 ± 0.00** |
| G4 | EfficientNet-B0, ImageNet, aug OFF | RGB | 100.00 ± 0.00 |
| G5 | EfficientNet-B0, ImageNet, two-stage | RGB | 100.00 ± 0.00 |
| G1g | Custom CNN, scratch | grayscale-eq | 84.41 ± 1.45 |
| G2g | EfficientNet-B0, scratch | grayscale-eq | 75.66 ± 2.38 |
| G3g | EfficientNet-B0, ImageNet | grayscale-eq | 100.00 ± 0.00 |

- **RGB is at the ceiling.** Every RGB model, including the 0.39 M-parameter
  baseline, scores 100%, as does a colour histogram with logistic regression. The
  proposal's +10 pp target for G3 over G1 is therefore **not met** (+0.00 pp).
- **Without colour, transfer learning wins.** G3g vs G1g is **+15.59 ± 1.45 pp**
  (paired bootstrap 95% CI excludes 0 for every seed). The whole gain comes from
  ImageNet initialisation: G3g vs G2g is +24.34 pp, and EfficientNet-B0 from scratch
  is 8.75 pp *worse* than the small CNN.
- **Augmentation only shows under shift.** G3 and G4 tie on the clean test set,
  but at the strongest occlusion G3 keeps 100.00% while G4 drops to 56.34%
  (`artifacts/metrics/robustness.md`; one final run per group, so no seed CI).
- **Colour dependence.** A grayscale probe drops G1 from 100% to 7.3% and G3 to
  67.4%.

Error analysis (`artifacts/error_analysis/`): the final G3, G3g and G1 models make no
test errors, so the analysis uses the final G1g model (64 errors out of 450). Six of
its ten analysed failures are corrected by showing the full frame instead of the
centre crop.

## Deviations from the proposal

| Item | Proposal | Implemented | Reason |
|---|---|---|---|
| Licence | "Academic Research Use" | CC BY-NC-SA 4.0 | Kaggle metadata |
| Citation | Winarno et al., ACM ICACS 2017 | Winarno et al. (2024), ACM JOCCH 17(2), doi:10.1145/3631131 | Citation requested by the dataset owner |
| Data size | ~3,000 RGB images at 224×224 | 3,020 iPhone 6s JPEGs (mostly 3024×4032); 2,993 after removing 27 exact duplicates | 224×224 is the model input, not the data |
| Split / leakage | Stratified split, pHash check | Group split by source photo: same EXIF second + rotation/flip-invariant pHash + ImageNet embedding similarity | A plain split leaks rotated copies of 98.9% of test images |
| Models | Custom CNN vs EfficientNet-B0 / ResNet-50 | Custom CNN, EfficientNet-B0 scratch, EfficientNet-B0 ImageNet | The scratch EfficientNet separates the architecture effect from the pretraining effect |
| Fine-tuning | Two-stage (linear probe, then unfreeze top blocks) | Matched single-stage for the headline; two-stage run as G5 | Avoids confounding the pretraining effect with a different schedule |
| Runs | 3 configs × 3 seeds + 3 sanity checks | G1–G4 × 3 seeds, plus G1g–G3g and G5; sanity checks reported separately | The ablation also needs several seeds |
| Success threshold | "+10% Macro F1" | +10 percentage points over the baseline (mean of 3 seeds) | "10%" was ambiguous between relative and absolute |
| Colour shortcut | Not anticipated | Grayscale + histogram-equalisation protocol and colour probes at evaluation | A colour histogram alone reaches 100% test Macro F1 |
| Compute | Colab T4, ~8–12 min per run (estimate) | Colab Tesla T4: 24 runs, 5.76 h of training, 9.2–31.0 min per run (`all_runs.csv`) | Measured, not estimated |
| Environment | Pinned PyTorch 2.5.1 | Training ran on Colab's PyTorch 2.11.0+cu130 / Python 3.13 (`artifacts/runs/*/env.json`). Re-aggregating those runs locally on PyTorch 2.5.1 gives byte-identical tables | Colab's preinstalled torch was used |


## Saved model

The final model is the G3 run with the best **validation** Macro F1
(`artifacts/metrics/final_model.json`). `python -m batik release` copies it to
`artifacts/release/batik_effb0_final.pt` (with `release.json`). That file and the
per-run `best.pt` checkpoints are published as assets of
[GitHub Release v1.0](https://github.com/Universitas-Cakrawala/semarang-batik-fine-grained-transfer-learning/releases/tag/v1.0):

| Asset | Contents | SHA-256 |
|---|---|---|
| `batik_checkpoints_v1.0.tar.gz` (283 MB) | `runs/<run_id>/best.pt` for all 24 runs + `release/batik_effb0_final.pt`; extract into `artifacts/` (`make checkpoints` does this) | `c4898ab9…94e7350` |
| `batik_effb0_final.pt` (16 MB) | Final model only, for `predict` and the demo | `8df9c65e…6133d8d1` |

They are not committed, because of their size and the dataset licence (weights
trained on CC BY-NC-SA data: non-commercial use only). Usage:

```bash
python -m batik predict path/to/photo.jpg --checkpoint artifacts/release/batik_effb0_final.pt
pip install -e ".[demo]" && make demo   # Gradio demo at http://127.0.0.1:7860
```

Online demo (same model, non-commercial):
[Hugging Face Space `atsiduy/vision-classification-batik`](https://huggingface.co/spaces/atsiduy/vision-classification-batik).

The model is a closed-set classifier over 10 motifs. It always returns one of
them, even for non-batik images, and a high confidence does not mean the image is
in scope (a chart image was labelled Asem Sinom with 0.81). Curators must verify
every prediction. It is not intended for deployment in museum or e-commerce systems.

## Data and licence

Semarang Batik Dataset, Kaggle `edywinarno/semarang-batik-dataset` v2, licensed
**CC BY-NC-SA 4.0**. Cite: Winarno, Septiarini, Hadikurniawati & Hamdani (2024),
*ACM Journal on Computing and Cultural Heritage* 17(2), doi:10.1145/3631131. See
[`DATA_LICENSE.md`](DATA_LICENSE.md) for what is and is not redistributed.

Code: MIT ([`LICENSE`](LICENSE)).

## Repository layout

```text
configs/            base.yaml + experiments/G*.yaml (only the keys that define each group)
src/batik/          data/ (download, audit, split, dataset, transforms), models, train,
                    metrics, sanity, diagnostics, analysis, robustness, errors, predict, cli
tests/              protocol unit tests (pytest)
notebooks/          Colab runner
data/manifests/     inventory, exclusions, split, leakage reports (committed; no pixels)
data/raw|processed/ dataset and resized cache (not committed)
artifacts/          runs/<run_id>/ (config, env, history, metrics, predictions, robustness), metrics/, sanity/,
                    error_analysis/, release/release.json (checkpoints: GitHub Release v1.0)
figures/            EDA, augmentation grid, curves, confusion matrix, error cases
reports/            report, contribution statement, AI-use declaration
demo/app.py         Gradio demo
```

## AI use

Implementation of the pipeline was assisted by Claude Code (Anthropic). The proposal
listed Gemini and GitHub Copilot. Each member's AI-use declaration in the report
must list every tool that member actually used. All reported numbers come from the
logged artefacts in `artifacts/`, not from any assistant.
