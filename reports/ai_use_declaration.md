# AI Use Declaration — per member

The proposal planned to use Google Gemini and GitHub Copilot (data-loader debugging,
plotting boilerplate, and LaTeX formatting). The pipeline implementation in this
repository was also assisted by **Claude Code (Anthropic)**. Each member declares
the tools they actually used.

Rules kept throughout (proposal Section 7):
- No metric is typed by hand or produced by an assistant. Every number traces to a
  file in `artifacts/` or `data/manifests/`.
- Causal error explanations are written by members after visually inspecting the
  failure evidence in `figures/errors/`.
- Every script was run and checked through the single-command, fixed-seed pipeline.

| Member | Tools used | What for | How the output was verified |
|---|---|---|---|
| Tita Noviana | Claude Code (Anthropic) | Asked it to check `reports/report_draft.md` §1–8 for traceability to `artifacts/`, and to cross-check §6 error-analysis claims (confidence, top-3, intervention results, sharpness) against `artifacts/error_analysis/G1g_customcnn-scratch_augon_s42/cases.json` and the 10 case images in `figures/errors/`; also used it to draft this AI-use entry and my contribution-statement row | I opened all 10 case images in `figures/errors/G1g_customcnn-scratch_augon_s42/` myself on 2026-10-28 and confirmed Claude's reading matched what I saw (crops cutting off the motif in cases 3/6/7, cases 8/9 being the same photo with two different wrong guesses, etc.) |
| Titanio Yudista | | | |
| Suci Fransisca Sisilia R. | Google Gemini / Antigravity | Assisted in cross-checking evaluation, ablation, and robustness tables (`reports/report_draft.md` §4–7) against `artifacts/metrics/{tables,robustness}.md` and `summary.json`, reviewing `src/batik/{models,train}.py`, and structuring the rewritten causal error analysis in §6 and my contribution/AI-use rows | Visually inspected all 10 failure figures in `figures/errors/G1g_customcnn-scratch_augon_s42/case_01–10.png` (Grad-CAM heatmaps, centre crop vs full frame, and nearest training neighbours) alongside `cases.json`, and ran `batik verify-split` and `batik aggregate` locally on 2026-10-29 to confirm all reported metrics and tables match byte-for-byte |
| Fajar Dwiharjo | Claude (Anthropic), chat interface | Asked it to check `reports/report_draft.md` against `artifacts/` and `figures/`, list every claim that does not match the files, and produce a corrected draft; also asked it to draft my row in the contribution statement and this entry, using the same findings. I did not ask it to write the causal interpretation in report section 6, which is still marked for rewrite by the member who owns error analysis. | Claims were checked on 2026-10-29 against the repo: table numbers against `tables.md` and `robustness.md`; split and leakage numbers against `inventory.csv`, `split_meta.json` and `near_duplicate_edges.csv`; error-case numbers against `cases.json`; `figures/confusion_matrix_final.png`, `case_03.png` and `case_08.png` viewed. |
