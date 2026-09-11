import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import os

BASE_DIR = os.path.dirname(__file__)

columns = []
columns += [f"mfcc_mean_{i}" for i in range(13)]
columns += [f"mfcc_std_{i}" for i in range(13)]
columns += [f"delta_std_{i}" for i in range(13)]
columns += [f"delta2_std_{i}" for i in range(13)]
columns += ["centroid_mean", "centroid_std", "bandwidth_mean", "bandwidth_std"]
columns += [f"contrast_mean_{i}" for i in range(7)]
columns += ["harmonic_ratio", "label"]

print("Loading CSVs...")
df_train = pd.read_csv(os.path.join(BASE_DIR, "train-norm.csv"), names=columns)
df_val = pd.read_csv(os.path.join(BASE_DIR, "val-norm.csv"), names=columns)

df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
df_val = df_val.sample(frac=1, random_state=42).reset_index(drop=True)

X = df_train.drop("label", axis=1).values
y = df_train["label"].values

X_val = df_val.drop("label", axis=1).values
y_val = df_val["label"].values

print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = RandomForestClassifier(n_estimators=600, max_depth=30, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42)
model.fit(X_train, y_train)

print("Validation Accuracy:", accuracy_score(y_val, model.predict(X_val)))
print("Test Accuracy:", accuracy_score(y_test, model.predict(X_test)))

joblib.dump(model, os.path.join(BASE_DIR, "audio_detection.pkl"))
print("Model saved to audio_detection.pkl")
