# Submission Checklist

This document lists the grading criteria and where to find the corresponding artifacts in the repository.

- [ ] Title / Author page (report)
  - Location: `project/report/report.tex`, compiled PDF: `results/report.pdf`
- [ ] Abstract, Introduction, Method, Experiments, Results, Discussion, References (report sections)
  - Location: `project/report/report.tex`
- [ ] Implementation (code) — runs without errors
  - Entry points: `python scripts/run_demo_pipeline.py`, `python src/train.py`
  - Supports Beta-VAE via `--beta` in `src/train.py`
- [ ] Reconstruction examples and visualization
  - Script: `scripts/reconstruct_and_visualize.py` (saves to `results/reconstructions/`)
- [ ] Clusterer comparison
  - Script: `scripts/cluster_comparison.py` (saves `results/cluster_comparison.csv` and `results/cluster_comparison.png`)
- [ ] Feature extraction & multimodality
  - Scripts: `src/features.py`, `scripts/lyrics_to_embeddings.py`, `src/prepare_multimodal.py`
- [ ] Correct evaluation metrics computation
  - Script: `src/eval_metrics.py` (used in pipeline)
  - Metrics summary: `results/metrics_summary.csv`
- [ ] Visualizations and figures
  - Figures included in `project/report/report.tex`, compiled to `results/report.pdf`
- [ ] Reproducibility
  - `requirements.txt` lists required packages
  - `setup_env.ps1` assists environment setup (PowerShell)
- [ ] Demo pipeline (optional)
  - `python scripts/run_demo_pipeline.py` produces demo outputs under `results/`

Notes:
- Use `project/tools/finalize_report.py --help` to generate a final report (insert metrics and compile).  
- Use `project/tools/clean_workspace.py --dry-run` to preview deletions before applying them.
