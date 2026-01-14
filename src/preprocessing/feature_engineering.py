import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = ['Amount','V1','V2','V3','V4','V5','V6']

def heuristic_labeling(df, amount_thresh=200, v1_thresh=-3):
    df = df.copy()
    df['heuristic_label'] = ((df['Amount'] > amount_thresh) | (df['V1'] < v1_thresh)).astype(int)
    return df

def extract_features(df, features=FEATURES, scaler=None):
    df_feat = df[features].fillna(0).copy()
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_feat)
    else:
        X_scaled = scaler.transform(df_feat)
    return X_scaled, scaler