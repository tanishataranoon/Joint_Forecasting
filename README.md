# Compound Drought + Heat Stress Forecasting — Thesis Codebase

An Explainable Multi-Branch Deep Learning Framework for Joint Drought and
Heat Stress Forecasting (Rajshahi region, Bangladesh). CSE 400 thesis.

This is the VS Code / script version of `thesis__1_.ipynb`, restructured so
training can run on a university GPU PC while everything else (data
pipeline, architecture dev, SHAP, LLM advisory) runs on a personal PC.

## Structure

```
configs/config.yaml     All hyperparameters, paths, dates, dims -- single source of truth
data/raw/                Raw NASA POWER + MODIS pulls (gitignored)
data/processed/           final_dataset.csv and friends (gitignored)
notebooks/                Exploration only -- import from src/, don't redefine functions here
src/data/fetch.py         NASA POWER + MODIS (GEE) fetching, temporal alignment
src/data/qc.py             Physical limits, IQR outliers, gap-aware interpolation
src/data/features.py      SPI-3, VPD, Humidex, severity label derivation
src/dataset.py             Sliding-window sequences, chronological split, scaling, DataLoader
src/models/branch_encoder.py   BranchEncoder (residual MLP per branch)
src/models/fusion.py       AdaptiveCrossAttentionFusion + BranchFusionBlock (core TC1 contribution)
src/models/heads.py        BiLSTM backbone + dual multi-task heads (SCAFFOLD -- not yet in notebook)
src/train.py                Training loop -- THIS runs on the university GPU
src/evaluate.py             Post-training metrics (macro-F1, ROC-AUC, regression)
src/explain.py               SHAP GradientExplainer wrapper (port from your v6 notebook Sec 13-14)
src/advisory.py              Gemini advisory layer (port from your v6 notebook Sec 15-16)
src/utils.py                 Seeding, device selection, checkpoint save/load
scripts/build_dataset.py    Runs the full data pipeline end-to-end -> final_dataset.csv
checkpoints/                  Model weights (gitignored, sync manually)
outputs/                      Figures, case-study reports
```

## One-time setup (both PCs)

```bash
python -m venv .venv
source .venv/bin/activate          # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env               # fill in GEMINI_API_KEY when you get to advisory.py
```

On the university PC, install the CUDA-matched PyTorch build instead of
the plain `torch` line in requirements.txt — check `nvidia-smi` first,
then use the selector at https://pytorch.org/get-started/locally/.

## Workflow

**Personal PC (now):**
1. `python scripts/build_dataset.py` — only needs to run once, wherever
   GEE is authenticated. Produces `data/processed/final_dataset.csv`.
2. Finish `src/models/heads.py` (backbone + heads aren't in the notebook
   yet) and sanity-check `src/train.py` on CPU with a tiny data slice —
   confirm shapes, no crashes, batch-size-1 compatibility.
3. `git push` the code. Copy `final_dataset.csv` to Drive/USB (too large
   for a normal git repo).

**University PC:**
1. `git pull`, `pip install -r requirements.txt` (+ CUDA torch).
2. Copy `final_dataset.csv` into `data/processed/`.
3. `python src/train.py --config configs/config.yaml`
4. Copy the `checkpoints/` folder back to Drive/USB when done.

**Personal PC (after):**
1. Copy `checkpoints/best_model.pt` into `checkpoints/`.
2. `python src/evaluate.py --checkpoint checkpoints/best_model.pt --split test`
3. `python src/explain.py --checkpoint checkpoints/best_model.pt` (SHAP)
4. Advisory layer / Bangla translation via `src/advisory.py`.

## Known TODOs (tracked, not forgotten)

- `src/models/heads.py`: finalize backbone + heads, matching what's
  currently only described (not yet coded) in your thesis notes.
- `src/explain.py` / `src/advisory.py`: port your working Section 13-16
  code from the v6 notebook rather than rewriting from scratch.
- `src/advisory_guideline_kb.json`: replace placeholder with real
  BRRI/BARC/BARI/DAE excerpts before final submission.
- Verify Humidex threshold (>=42°C) against BRRI/FAO Boro crop-stage
  sensitivity sources.
- Ablation study script for the BiLSTM backbone design choice.
