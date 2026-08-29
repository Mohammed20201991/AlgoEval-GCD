# AlgoEval-GCD

**Intelligent Cloud Resource Usage Prediction to Improve Task Scheduling using Machine Learning**

This repository contains the code used to produce the results reported in
*[paper title], F1000Research 15:366*. It evaluates supervised learning models for
predicting task-level CPU usage on the Google Cluster Data v3 trace, using only
information available to the scheduler at task dispatch time.

## Repository structure

- **`notebooks/cpu_prediction_pipeline.ipynb`** — the canonical, single-source-of-truth
  pipeline. Running it end to end reproduces every table and figure in the paper
  (dataset report, Table 2; regression results, Table 3; classification results, Table 4;
  Figures 1, 5, 6, 7).
- **`results/tables/`** — exported CSVs of the tables above.
- **`results/figures/`** — exported figure images.
- **`legacy/`** — earlier exploratory scripts from initial development. Superseded by the
  notebook above; kept for historical reference only. **Do not use these for reproducing
  paper results.**

## Data

The pipeline uses the Kaggle sample of the Google Cluster Trace v3
(`derrickmwiti/google-2019-cluster-sample`, `borg_traces_data.csv`), downloaded
automatically via `kagglehub` on first run and cached locally / to Google Drive on
subsequent runs. See [google/cluster-data](https://github.com/google/cluster-data) for the
full trace and its official documentation.

## Methodology summary

- **Target**: realized average CPU utilization during a task's execution.
- **Features**: dispatch-time-only fields (scheduling class, priority, requested CPU/memory,
  scheduler, vertical-scaling policy, collection type), plus engineered features (resource
  ratios, constraint counts, event-type codes, and historical per-collection usage features
  computed only from earlier task instances).
- **Leakage guard**: no field derived from a task's own execution (realized usage, execution
  timing, derived usage statistics) is ever used as a model input — only dispatch-time
  information is used to predict outcomes.
- **Models**: Linear Regression, Support Vector Regression, Random Forest, and a Neural
  Network for regression; Logistic Regression, SVC, Random Forest, and a Neural Network for
  a separate, secondary high-CPU classification task.
- **Splits**: 70% / 15% / 15% train / validation / test, evaluated identically across all
  models.

## Usage

```bash
git clone https://github.com/Mohammed20201991/AlgoEval-GCD.git
cd AlgoEval-GCD
pip install -r requirements.txt
jupyter notebook notebooks/cpu_prediction_pipeline.ipynb
```

Run all cells top to bottom. The notebook prints the exact dataset size and train/val/test
split sizes (quoted verbatim in the paper's Methods), followed by the regression and
classification results tables.

## Results

See the paper for full discussion. Headline test-set results:

| Task | Best model | Key metric |
|---|---|---|
| Regression | RF + Extra Trees ensemble | R² ≈ 0.895, MAE ≈ 0.0023 |
| Classification | Random Forest | Accuracy ≈ 0.921, F1 ≈ 0.844 |

## Citation

```bibtex
@misc{algoevalgcd2025,
  title   = {AlgoEval-GCD: Algorithm Evaluation on Google Cluster Data},
  author  = {Al-Hitawi, Mohammed A.S. and Hadi, Ahmed},
  year    = {2025},
  url     = {https://github.com/Mohammed20201991/AlgoEval-GCD}
}
```

## License

MIT License — see `LICENSE`.