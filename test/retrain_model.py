"""
Retrain the AI Audio Detector model.

Usage:
    uv run python retrain_model.py --dataset /path/to/for-norm/for-norm

The dataset should have subfolders: training/, validation/, testing/
Each with 'real/' and 'fake/' subdirectories containing .wav/.mp3 files.

Download dataset from:
https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset
"""

import os
import csv
import sys
import argparse
import numpy as np
import librosa
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def extract_features(y, sr):
    try:
        features = []
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features.extend(np.std(delta, axis=1))
        features.extend(np.std(delta2, axis=1))
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features.append(np.mean(centroid))
        features.append(np.std(centroid))
        features.append(np.mean(bandwidth))
        features.append(np.std(bandwidth))
        features.extend(np.mean(contrast, axis=1))
        y_harm, y_perc = librosa.effects.hpss(y)
        harmonic_ratio = np.mean(np.abs(y_harm)) / (np.mean(np.abs(y_perc)) + 1e-9)
        features.append(harmonic_ratio)
        return features
    except Exception as e:
        print(f"  [skip] {e}")
        return None


def extract_split(split_path, out_csv):
    total = 0
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        for label_name, label_id in [("real", 0), ("fake", 1)]:
            folder = os.path.join(split_path, label_name)
            if not os.path.isdir(folder):
                print(f"  [warn] Folder not found: {folder}")
                continue
            files = [x for x in os.listdir(folder) if x.lower().endswith((".wav", ".mp3"))]
            print(f"  Processing {len(files)} {label_name} files...")
            for i, fname in enumerate(files, 1):
                path = os.path.join(folder, fname)
                try:
                    y, sr = librosa.load(path, sr=16000, mono=True)
                    feats = extract_features(y, sr)
                    if feats is not None:
                        writer.writerow(feats + [label_id])
                        total += 1
                except Exception as e:
                    print(f"  [skip] {fname}: {e}")
                if i % 100 == 0:
                    print(f"    {i}/{len(files)} done...")
    print(f"  Saved {total} rows to {out_csv}")


def get_columns():
    cols = []
    cols += [f"mfcc_mean_{i}" for i in range(13)]
    cols += [f"mfcc_std_{i}" for i in range(13)]
    cols += [f"delta_std_{i}" for i in range(13)]
    cols += [f"delta2_std_{i}" for i in range(13)]
    cols += ["centroid_mean", "centroid_std", "bandwidth_mean", "bandwidth_std"]
    cols += [f"contrast_mean_{i}" for i in range(7)]
    cols += ["harmonic_ratio", "label"]
    return cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="Path to dataset root (contains training/, validation/, testing/)")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip feature extraction if CSVs already exist")
    args = parser.parse_args()

    train_csv, val_csv, test_csv = "train-norm.csv", "val-norm.csv", "test-norm.csv"

    if not args.skip_extraction:
        print("\n[1/3] Extracting training features...")
        extract_split(os.path.join(args.dataset, "training"), train_csv)
        print("\n[1/3] Extracting validation features...")
        extract_split(os.path.join(args.dataset, "validation"), val_csv)
        print("\n[1/3] Extracting testing features...")
        extract_split(os.path.join(args.dataset, "testing"), test_csv)
    else:
        print("[1/3] Skipping extraction (using existing CSVs)")

    print("\n[2/3] Loading data...")
    cols = get_columns()
    df_train = pd.read_csv(train_csv, names=cols).sample(frac=1, random_state=42)
    df_val   = pd.read_csv(val_csv,   names=cols).sample(frac=1, random_state=42)

    X_train = df_train.drop("label", axis=1).values
    y_train = df_train["label"].values
    X_val   = df_val.drop("label", axis=1).values
    y_val   = df_val["label"].values
    print(f"  Train: {len(X_train)} samples | Val: {len(X_val)} samples")

    print("\n[3/3] Training RandomForest (this may take a few minutes)...")
    model = RandomForestClassifier(
        n_estimators=600, max_depth=30, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    val_acc = accuracy_score(y_val, model.predict(X_val))
    print(f"  Validation Accuracy: {val_acc:.4f}")

    joblib.dump(model, "audio_detection.pkl")
    print("\nDone! Model saved to audio_detection.pkl")
    print("Now run:  uv run python main.py")


if __name__ == "__main__":
    main()
