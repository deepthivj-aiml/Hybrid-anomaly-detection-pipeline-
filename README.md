# Anomaly-Detection

Hybrid anomaly detection pipeline combining an autoencoder and a supervised "judge" (XGBoost).

Structure
- notebooks/: exploratory notebooks (Colab-friendly)
- data/: raw and processed datasets
- src/: reusable library modules (data, preprocessing, models, training, evaluation)
- scripts/: helper scripts to run pipeline
- .github/workflows/: CI for tests / linting

Quickstart
1. Create a virtual environment and install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the pipeline:
   ```
   bash scripts/run_pipeline.sh
   ```
3. Artifacts (models, scaler) are written to `artifacts/` by default.

License: Add a license file (e.g., MIT) if you want the repo to be open-source.