# kaim-week3-insurance-risk-analytics-acis

End-to-End Insurance Risk Analytics & Predictive Modeling for AlphaCare Insurance Solutions (ACIS)

## Overview
Analytics pipeline to explore historical car insurance data (Feb 2014–Aug 2015), quantify risk, and support pricing/marketing decisions. Key pieces:
- Reproducible data/version control via DVC
- Automated analytics pipeline in `src/pipeline.py`
- Data loader and cleaning utilities in `src/data_loader.py`
- Reports in `reports/` and notebooks in `notebooks/`

## Environment setup
```bash
conda create -n kaim_week3 python=3.11 -y
conda activate kaim_week3
pip install -r requirements.txt
```

## Data with DVC
Data is tracked with DVC using a local remote configured at `/home/nabi/dvc-storage/insurance-project`.

Pull tracked artifacts (processed parquet + raw txt):
```bash
dvc pull
```

If you change data, track and push back:
```bash
dvc add data/raw/MachineLearningRating_v3.txt
dvc add data/processed/insurance_data_final_cleaned.parquet
git add data/raw/MachineLearningRating_v3.txt.dvc data/processed/insurance_data_final_cleaned.parquet.dvc .gitignore
git commit -m "Track data with DVC"
dvc push
```

## Run the analytics pipeline
```bash
python scripts/run_analysis.py
```
Defaults read from `data/processed/insurance_data_final_cleaned.parquet`; override by passing a path to `InsuranceAnalyticsPipeline(data_path=...)` in `scripts/run_analysis.py` if needed. Output report saves to `reports/automated_analysis_report.md`.

## Hypothesis testing (Task-3)
Run statistical tests for provinces, zip codes, margin, and gender:
```bash
python scripts/run_hypothesis_tests.py \
  --data-path data/processed/insurance_data_final_cleaned.parquet \
  --report-path reports/hypothesis_testing.md
```
Console output shows reject/fail decisions; full Markdown report is saved to `reports/hypothesis_testing.md`.

## Tests and CI
```bash
pytest -v
```
GitHub Actions runs `pytest` on pushes to `main` and `task-*` (see `.github/workflows/unittests.yml`).

## Branch/PR cadence (week tasks)
- Task-1 (EDA/stats) and Task-2 (DVC) work live on feature branches (`task-1`, `task-2`) and merge to `main` via PRs.
- Current branch: `task-3` (hypothesis testing).

## Repo structure (high level)
```
src/                # core pipeline and loaders
scripts/run_analysis.py
data/raw/           # raw data (DVC tracked)
data/processed/     # processed parquet (DVC tracked)
notebooks/          # exploratory notebooks
reports/            # generated reports
tests/              # basic DVC/notebook presence checks
```

## Notes and next steps
- If `dvc pull` fails, ensure the local remote exists and is accessible.
- Add richer tests for data validity and metrics calculations.
- Extend the report with hypothesis testing and modeling outputs for Task-3/4.
