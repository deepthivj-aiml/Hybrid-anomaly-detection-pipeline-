import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from joblib import dump
from ..data.load_data import load_sample
from ..preprocessing.feature_engineering import heuristic_labeling, extract_features
from ..models.autoencoder import Autoencoder
from ..models.judge import create_xgb_classifier
from ..config import AE_EPOCHS, RANDOM_STATE

def run_pipeline(sample_size=50000, save_dir="artifacts"):
    os.makedirs(save_dir, exist_ok=True)

    # Load and label
    df = load_sample(n_samples=sample_size)
    df = heuristic_labeling(df)

    # Features
    X_scaled, scaler = extract_features(df)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Train AE
    ae = Autoencoder(X_scaled.shape[1])
    opt = torch.optim.Adam(ae.parameters(), lr=0.01)
    crit = nn.MSELoss()

    print("Training Autoencoder...")
    for epoch in range(AE_EPOCHS):
        opt.zero_grad()
        loss = crit(ae(X_tensor), X_tensor)
        loss.backward()
        opt.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Anomaly scores
    with torch.no_grad():
        recon = ae(X_tensor)
        errors = torch.mean((recon - X_tensor)**2, dim=1).numpy()

    df['ae_score'] = errors
    df['ae_score_scaled'] = (errors - errors.mean()) / (errors.std() + 1e-8)

    # Train judge
    X_final = np.hstack([X_scaled, df['ae_score_scaled'].values.reshape(-1,1)])
    y = df['heuristic_label'].values
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=RANDOM_STATE)

    print("Training XGBoost judge...")
    model = create_xgb_classifier()
    model.fit(X_train, y_train)

    # Save artifacts
    dump(model, os.path.join(save_dir, "xgb_judge.joblib"))
    torch.save(ae.state_dict(), os.path.join(save_dir, "autoencoder.pt"))
    dump(scaler, os.path.join(save_dir, "scaler.joblib"))
    print("Artifacts saved to", save_dir)

    return model, ae, scaler, X_test, y_test

if __name__ == "__main__":
    run_pipeline()