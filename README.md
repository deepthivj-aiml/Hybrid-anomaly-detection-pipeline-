# Domain-Driven Hybrid Anomaly Detection for Streaming Behavior

![Python](https://img.shields.io/badge/python-3.10-blue) ![PyTorch](https://img.shields.io/badge/pytorch-2.0-orange) ![XGBoost](https://img.shields.io/badge/xgboost-1.7-red)

## 📖 Project Overview
This repository demonstrates a domain-driven, Netflix-inspired approach to anomaly detection that simulates streaming behavior using an open credit-card dataset. The core idea: combine domain heuristics with semi-supervised and supervised models to build a practical, production-minded anomaly detection flow.

Key components:
- Domain-driven heuristics / rules to bootstrap labels
- Semi-supervised Autoencoder to produce reconstruction-based anomaly scores and embeddings
- Supervised "judge" (XGBoost) that combines engineered features + autoencoder outputs to produce final anomaly probabilities

This hybrid approach aims to improve detection accuracy and reduce false positives compared to heuristics-only systems.

---

## 🎯 Objectives
1. Simulate streaming fraud/anomaly detection with public data.  
2. Bootstrap labels with domain heuristics.  
3. Engineer behavioral features rather than relying only on raw stats.  
4. Train a semi-supervised Autoencoder for anomaly scoring.  
5. Use XGBoost as a supervised judge combining embeddings, recon errors and features.  
6. Evaluate detection effectiveness (precision/recall/F1/AUC etc.).

---

## 🗂️ Dataset (demo)
- Source: Credit Card Fraud Detection dataset (OpenML, `fetch_openml("creditcard")`).  
- Demo subset: 50,000 rows (configurable) for quick experiments.  
- Example features used: `Amount`, `V1`–`V6` (numeric PCA-like components in original dataset).  
- Heuristic pseudo-labels simulate streaming “unexpected behavior” (e.g., very large amounts, unusual combinations of V-features).

---

## ⚙️ Pipeline Overview

1. Data loading (OpenML or local CSV/Parquet)  
2. Domain heuristics create pseudo-labels for bootstrapping  
3. Feature engineering (scaling, rolling stats / session summaries where applicable)  
4. Train Autoencoder (semi-supervised) on normal-ish data to learn reconstruction; compute recon error + embeddings  
5. Train XGBoost judge using features + autoencoder outputs  
6. Evaluate end-to-end: AUC-ROC, Precision@k, F1, confusion matrices, error histograms

---

## 🔧 Tech Stack
- Python 3.10
- PyTorch (Autoencoder)
- XGBoost (Judge)
- pandas, numpy, scikit-learn, matplotlib, seaborn, openml
- Colab-friendly notebooks included

---

## Repository structure
- `notebooks/` — demos and Colab-friendly walkthroughs  
- `src/` — core modules (data, preprocessing, models, training, evaluation)  
- `scripts/` — convenience scripts (data download, run pipeline, training steps)  
- `config/` — YAML configs for experiments  
- `data/` — raw & processed datasets (not checked in)  
- `artifacts/` — default output (trained models, scalers, reports)  
- `.github/workflows/` — CI (tests & lint)

---

## Quickstart

1. Clone the repo
```bash
git clone https://github.com/deepthivj-aiml/Hybrid-anomaly-detection-pipeline-.git
cd Hybrid-anomaly-detection-pipeline-
```

2. Create & activate virtualenv, then install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
.venv\Scripts\activate       # Windows (PowerShell)
pip install -r requirements.txt
```

3. Download demo subset from OpenML (example script provided):
```bash
python scripts/download_openml_subset.py --dataset creditcard --n_rows 50000 --out data/raw/creditcard_50k.csv
```

4. Run the end-to-end demo:
```bash
bash scripts/run_pipeline.sh --config config/config.yaml
```
Or run the steps individually:
```bash
python scripts/train_autoencoder.py --config config/autoencoder.yaml
python scripts/generate_embeddings.py --config config/infer.yaml
python scripts/train_judge.py --config config/judge.yaml
python scripts/evaluate.py --config config/eval.yaml
```

---

## Configuration
Pipeline behavior is controlled via YAML files in `config/`. Typical sections:
- `data`: raw/processed paths, sample size, target column
- `preprocessing`: scaler type, columns to use
- `autoencoder`: model arch (latent dim), epochs, batch_size, learning_rate, seed
- `judge`: XGBoost params (n_estimators, max_depth, learning_rate), early_stopping
- `artifacts`: output directory, filenames

Example:
```yaml
data:
  raw_path: data/raw/creditcard_50k.csv
  processed_dir: data/processed
  target_col: is_anomaly
autoencoder:
  latent_dim: 16
  epochs: 50
  batch_size: 128
  learning_rate: 1e-3
judge:
  n_estimators: 200
  max_depth: 6
  learning_rate: 0.1
artifacts:
  output_dir: artifacts/
```

---

## Data format & recommended layout
- Tabular CSV or Parquet, rows = observations, columns = features.  
- If labels exist: binary column `is_anomaly` with 1 = anomaly, 0 = normal.  
- For streaming simulation, include a timestamp or session id column if you want to compute behavioral aggregations.

Recommended:
```
data/
  raw/
    creditcard_50k.csv
  processed/
    train.parquet
    val.parquet
    test.parquet
```

---

## Feature engineering suggestions
- Standard scaling for numeric features  
- Add engineered behavioral features:
  - rolling mean/std over last N events (if timestamped)
  - session totals / averages
  - time-since-last-event
- Autoencoder: use embeddings and reconstruction error (L1/L2) as features for the judge

---

## Training & evaluation details
- Autoencoder training: minimize reconstruction loss on mostly-normal data. Use recon error as anomaly score.
- Judge training: supervised XGBoost that combines features + recon error + embeddings.
- Metrics:
  - AUC-ROC, Average Precision
  - Precision@k (top-k anomalies)
  - Precision, Recall, F1 at selected thresholds
- Visuals:
  - Histograms of reconstruction errors
  - Precision/Recall curves
  - Feature importance from XGBoost

Reproducibility: set `seed` in configs so numpy, torch, and random are fixed.

---

## Notebooks & demos
- `notebooks/` contains:
  - A Colab-friendly demo that downloads OpenML data, creates heuristics, trains the autoencoder, trains the judge, and evaluates results.
  - Exploratory notebooks for feature design and error analysis.

---

## Artifacts
Default outputs are written to `artifacts/`:
- `artifacts/autoencoder/` — model checkpoints, training logs
- `artifacts/judge/` — XGBoost model dumps, feature pipeline metadata
- `artifacts/features/` — datasets augmented with embeddings and recon errors
- `artifacts/reports/` — metrics, plots, and evaluation reports

---

## Development, testing & CI
- Unit tests:
```bash
pytest
```
- Linting & formatting:
```bash
flake8 src
black .
```
- CI: see `.github/workflows/` for lint & test jobs.

---

## Practical recommendations for streaming
- Use heuristics to flag suspicious events and collect human feedback to grow labeled data.
- Use the autoencoder scores for continuous monitoring and to prioritize events for labeling.
- Retrain judge periodically as more labeled examples arrive; consider online / incremental learning for judge if data changes fast.
- Monitor model drift (score distributions) and data quality.

---

## Contributing
Contributions welcome:
1. Open an issue describing your proposal/bug.
2. Create a branch off `main`.
3. Add tests for new functionality.
4. Submit a PR with a clear description and changelog entry.

Please follow existing code style and include reproducible examples for new features.

---

## License
Add a `LICENSE` file for the repository. Suggested: MIT License for an open-source demo.

---

## Contact
Maintainer: deepthivj-aiml (GitHub)  
For questions, issues, or collaboration ideas, please open an issue in this repository.

---

## Citation
If you use this pipeline in research or demos, please cite the repository and include a link to the commit used.
