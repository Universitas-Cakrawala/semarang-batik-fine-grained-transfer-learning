# AI Use Declaration — per member

The proposal planned to use Google Gemini and GitHub Copilot (data-loader debugging,
plotting boilerplate, and LaTeX formatting). The pipeline implementation in this
repository was also assisted by **Claude Code (Anthropic)**. Each member declares
the tools they actually used.

Rules kept throughout (proposal Section 7):
- No metric is typed by hand or produced by an assistant. Every number traces to a
  file in `artifacts/` or `data/manifests/`.
- Causal error explanations in report §6 were rewritten by Suci after visually inspecting the
  failure evidence in `figures/errors/`. Later corrections to individual visual claims (cases 1, 2,
  3, 4, 7 and 9, and the sharpness caveat) were made with Claude Code at Titanio's request and
  checked against the same figures and `cases.json`.
- Every script was run and checked through the single-command, fixed-seed pipeline.

| Member | Tools used | What for | How the output was verified |
|---|---|---|---|
| Tita Noviana | Claude Code (Anthropic) | Asked it to check `reports/report_draft.md` §1–8 for traceability to `artifacts/`, and to cross-check §6 error-analysis claims (confidence, top-3, intervention results, sharpness) against `artifacts/error_analysis/G1g_customcnn-scratch_augon_s42/cases.json` and the 10 case images in `figures/errors/`; also used it to draft this AI-use entry and my contribution-statement row | I opened all 10 case images in `figures/errors/G1g_customcnn-scratch_augon_s42/` myself on 2026-10-28 and confirmed Claude's reading matched what I saw (crops cutting off the motif in cases 3/6/7, cases 8/9 being the same photo with two different wrong guesses, etc.) |
| Titanio Yudista | Claude Code (Anthropic) | Implementing the pipeline in `src/batik/`, `configs/`, `tests/`, `Makefile`, `notebooks/colab_runner.ipynb` and `demo/app.py`; debugging the Colab runs (DataLoader worker warning, empty error-analysis fallback, stale code without `git pull`); preparing GitHub Release v1.0 and `make checkpoints`; deploying the Hugging Face Space; filling `reports/report_draft.md` §4–8 from `artifacts/`, correcting visual claims in §6 against `figures/errors/`, and drafting my contribution row and this entry | `make test` (15 passed); `make reproduce` = 100.00 ± 0.00 % on my laptop and from a fresh clone with the released checkpoints; re-running `batik aggregate` locally gave byte-identical `tables.md`, `all_runs.csv`, `group_summary.csv`, `summary.json` and `final_model.json`; the demo was tested through its API (10 of 10 test photos correct); report numbers checked against `artifacts/metrics/*`, `data/manifests/*` and `artifacts/runs/*/history.csv` |
| Suci Fransisca Sisilia R. | Google Gemini / Antigravity | Assisted in cross-checking evaluation, ablation, and robustness tables (`reports/report_draft.md` §4–7) against `artifacts/metrics/{tables,robustness}.md` and `summary.json`, reviewing `src/batik/{models,train}.py`, and structuring the rewritten causal error analysis in §6 and my contribution/AI-use rows | Visually inspected all 10 failure figures in `figures/errors/G1g_customcnn-scratch_augon_s42/case_01–10.png` (Grad-CAM heatmaps, centre crop vs full frame, and nearest training neighbours) alongside `cases.json`, and ran `batik verify-split` and `batik aggregate` locally on 2026-10-29 to confirm all reported metrics and tables match byte-for-byte |
| Fajar Dwiharjo | Claude (Anthropic), chat interface | Asked it to check `reports/report_draft.md` against `artifacts/` and `figures/`, list every claim that does not match the files, and produce a corrected draft; also asked it to draft my row in the contribution statement and this entry, using the same findings. I did not ask it to write the causal interpretation in report section 6, which is still marked for rewrite by the member who owns error analysis. | Claims were checked on 2026-10-29 against the repo: table numbers against `tables.md` and `robustness.md`; split and leakage numbers against `inventory.csv`, `split_meta.json` and `near_duplicate_edges.csv`; error-case numbers against `cases.json`; `figures/confusion_matrix_final.png`, `case_03.png` and `case_08.png` viewed. |
