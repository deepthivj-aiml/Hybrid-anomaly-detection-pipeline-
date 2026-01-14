import os
import urllib.request
import pandas as pd
from ..config import DATA_URL, RAW_DATA_PATH, RANDOM_STATE

def download_dataset(force=False):
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    if not os.path.exists(RAW_DATA_PATH) or force:
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATA_URL, RAW_DATA_PATH)
    else:
        print("Dataset already present:", RAW_DATA_PATH)

def load_sample(n_samples=50000, random_state=RANDOM_STATE):
    download_dataset()
    df = pd.read_csv(RAW_DATA_PATH)
    if n_samples and n_samples < len(df):
        df = df.sample(n_samples, random_state=random_state).reset_index(drop=True)
    return df