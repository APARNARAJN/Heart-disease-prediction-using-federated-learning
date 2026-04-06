"""
Run this script ONCE from the same folder as your cardio_train.csv
It will recreate scaler.pkl without re-running the full FL training.

Usage:
    python recreate_scaler.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

# Load the same dataset
df = pd.read_csv('cardio_train.csv', delimiter=';')
if 'id' in df.columns:
    df = df.drop('id', axis=1)

print(f"✓ Dataset loaded: {len(df):,} records")

# Fit scaler on all features (same as training script)
all_X = df.drop('cardio', axis=1).values
scaler = StandardScaler().fit(all_X)

# Save it
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✓ scaler.pkl saved successfully!")
print("  You can now run: streamlit run fl_ui.py")