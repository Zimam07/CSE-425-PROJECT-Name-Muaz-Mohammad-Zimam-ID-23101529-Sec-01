# VAE for Hybrid Language Music Clustering

This repository contains the code and scripts to reproduce the Unsupervised Learning project: VAE-based clustering for hybrid audio+lyrics music datasets.

Overview
- Dataset: Jamendo (Kaggle) as primary dataset (audio previews + metadata/lyrics). Can be supplemented with Kaggle lyrics datasets for Bangla/Hindi coverage.
- Framework: PyTorch (recommended)

Quick start
1. Install Python 3.8+.
2. From project root, run (PowerShell):
   ```powershell
   .\setup_env.ps1
   ```
3. Put your Kaggle API key (`kaggle.json`) in `%USERPROFILE%\.kaggle\kaggle.json` if you want to use the Kaggle API.
4. Use `python scripts/download_jamendo.py` to download or follow the instructions to download manually. Alternatively, to import lyrics-only Kaggle datasets (e.g. `neisse/scrapped-lyrics-from-6-genres`) using `kagglehub`, run:

```powershell
python scripts/load_kagglehub_lyrics.py --dataset neisse/scrapped-lyrics-from-6-genres --write_texts
```

This will create `data/raw/metadata.csv` and `data/raw/lyrics/*.txt` (and `data/raw/metadata_aligned.csv`).
4. Extract audio features: `python src/features.py --input_dir data/raw/audio --output_dir data/features` (see script docstring)
5. Validate and align metadata/lyrics: `python scripts/validate_dataset.py --audio_dir data/raw/audio --lyrics_dir data/raw/lyrics --metadata data/raw/metadata.csv --out data/raw/metadata_aligned.csv`
6. Generate lyrics embeddings: `python scripts/lyrics_to_embeddings.py --metadata data/raw/metadata_aligned.csv --out_dir data/features/lyrics_embeddings --model all-mpnet-base-v2`
7. Prepare multimodal concatenated features (audio + lyrics): `python src/prepare_multimodal.py --audio_dir data/features --lyrics_dir data/features/lyrics_embeddings --out_dir data/features/multimodal`
8. Quick clustering evaluation (PCA+KMeans + metrics): `python scripts/quick_eval.py --feat_dir data/features/multimodal --k 10 --out results/metrics.csv`
9. Train a basic VAE: `python src/train.py --feat_dir data/features --out_dir results`.

Structure
```
project/
  data/
    raw/            # raw audio + lyrics (not committed)
    features/       # extracted MFCC / spectrograms (.npy)
  notebooks/        # exploratory notebooks
  scripts/
    download_jamendo.py
  src/
    features.py
    dataset.py
    vae.py
    train.py
  results/
  requirements.txt
  README.md
```

Notes
- Start with the Easy task: basic VAE on MFCC or spectrogram features, KMeans clustering on latent space, and compare with PCA+KMeans.
- For lyrics embedding we recommend `sentence-transformers` (SBERT) for quick, high-quality text embeddings.

If you want, I can start the data download and feature extraction now (I will need Kaggle credentials or you can provide the Jamendo files).

Report and deliverables
-----------------------
I prepared a demo run and generated a draft report and notebook. To reproduce the demo report locally:

```powershell
cd project
python scripts/run_demo_pipeline.py    # creates demo data and runs pipeline
python scripts/generate_report_parts.py --metrics results/demo_analysis/metrics.csv --notebook notebooks/latent_analysis.ipynb --out_dir results/report_parts
```

Outputs produced in the demo run:
- `results/demo_vae/` (VAE checkpoints and `latents.npy`)
- `results/demo_analysis/` (metrics and plots)
- `results/report_parts/latent_analysis.html` (notebook exported to HTML)
- `report/report.tex` (LaTeX draft referencing metrics and figures)

To build a PDF from `report/report.tex` you can run a LaTeX toolchain (TeXLive/MiKTeX):

```powershell
pdflatex -output-directory report report/report.tex
```

If you prefer, I can export the notebook outputs and assemble a short PDF report for you.


## Finalize and submission helpers

- `python project/tools/finalize_report.py --analysis <path> --compile`  — insert metrics from `results/metrics_summary.csv` into the demo table and compile the report (uses bundled tectonic).
- `python project/tools/clean_workspace.py --dry-run` — preview candidate deletions. To actually delete, run `python project/tools/clean_workspace.py --yes` (careful: will remove intermediate build files).
- `build.ps1` — convenience wrapper for Windows: `.uild.ps1 -RunDemo -Compile` will run demo then compile.

See `SUBMISSION_CHECKLIST.md` at the repository root for the final submission checklist and mappings to artifacts.

Demo: self-contained pipeline
----------------------------
If you don't have dataset files yet or just want to validate the pipeline, run the included demo which creates synthetic audio + lyrics and executes the full pipeline end-to-end (validation, feature extraction, embeddings, multimodal preparation, and quick evaluation):

    python scripts/run_demo_pipeline.py

The demo is useful to verify the entire workflow without external downloads and produces example metrics at `results/demo_metrics.csv`. Feel free to run it now and inspect the outputs.