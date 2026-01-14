import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("--- Model Performance Report ---")
    print(classification_report(y_test, y_pred))
    return y_pred

def plot_ae_errors(errors):
    plt.figure(figsize=(10,6))
    plt.hist(errors, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(np.percentile(errors, 95), color='red', linestyle='--', label='95th Percentile')
    plt.title("Autoencoder Reconstruction Error Distribution")
    plt.xlabel("Reconstruction Error (Anomaly Score)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()