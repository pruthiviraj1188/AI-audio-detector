#!/usr/bin/env python3
"""
Check the accuracy of the trained model on all datasets
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

BASE_DIR = os.path.dirname(__file__)

# Define column names (61 features + label)
columns = []
columns += [f"mfcc_mean_{i}" for i in range(13)]
columns += [f"mfcc_std_{i}" for i in range(13)]
columns += [f"delta_std_{i}" for i in range(13)]
columns += [f"delta2_std_{i}" for i in range(13)]
columns += ["centroid_mean", "centroid_std", "bandwidth_mean", "bandwidth_std"]
columns += [f"contrast_mean_{i}" for i in range(7)]
columns += ["harmonic_ratio", "label"]

print("=" * 70)
print("🎯 AI AUDIO DETECTOR - MODEL ACCURACY CHECK")
print("=" * 70)

# Load model
print("\n📦 Loading model...")
model_path = os.path.join(BASE_DIR, "audio_detection.pkl")
model = joblib.load(model_path)
print(f"✅ Model loaded: {type(model).__name__}")
print(f"   Estimators: {model.n_estimators}")
print(f"   Max Depth: {model.max_depth}")

# Load datasets
print("\n📊 Loading datasets...")
df_train = pd.read_csv(os.path.join(BASE_DIR, "train-norm.csv"), names=columns)
df_val = pd.read_csv(os.path.join(BASE_DIR, "val-norm.csv"), names=columns)
df_test = pd.read_csv(os.path.join(BASE_DIR, "test-norm.csv"), names=columns)

print(f"   Training samples: {len(df_train):,}")
print(f"   Validation samples: {len(df_val):,}")
print(f"   Test samples: {len(df_test):,}")
print(f"   Total samples: {len(df_train) + len(df_val) + len(df_test):,}")

# Prepare data
X_train = df_train.drop("label", axis=1).values
y_train = df_train["label"].values

X_val = df_val.drop("label", axis=1).values
y_val = df_val["label"].values

X_test = df_test.drop("label", axis=1).values
y_test = df_test["label"].values

# Calculate accuracies
print("\n" + "=" * 70)
print("📈 ACCURACY RESULTS")
print("=" * 70)

# Training accuracy
y_train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)
print(f"\n🔵 Training Accuracy:   {train_acc * 100:.2f}%")

# Validation accuracy
y_val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"🟢 Validation Accuracy: {val_acc * 100:.2f}%")

# Test accuracy
y_test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"🟡 Test Accuracy:       {test_acc * 100:.2f}%")

# Detailed classification report for test set
print("\n" + "=" * 70)
print("📋 DETAILED CLASSIFICATION REPORT (Test Set)")
print("=" * 70)
print("\nClass Labels: 0 = Human, 1 = AI Generated")
print("\n" + classification_report(y_test, y_test_pred, 
                                    target_names=["Human Audio", "AI Generated"],
                                    digits=4))

# Confusion matrix
print("\n" + "=" * 70)
print("🔢 CONFUSION MATRIX (Test Set)")
print("=" * 70)
cm = confusion_matrix(y_test, y_test_pred)
print("\n                Predicted")
print("              Human    AI")
print(f"Actual Human   {cm[0][0]:5d}  {cm[0][1]:5d}")
print(f"       AI      {cm[1][0]:5d}  {cm[1][1]:5d}")

# Calculate per-class accuracy
human_acc = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
ai_acc = cm[1][1] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0

print(f"\nHuman Audio Detection Accuracy: {human_acc * 100:.2f}%")
print(f"AI Audio Detection Accuracy:    {ai_acc * 100:.2f}%")

# Feature importance (top 10)
print("\n" + "=" * 70)
print("🎯 TOP 10 MOST IMPORTANT FEATURES")
print("=" * 70)
feature_names = columns[:-1]  # Exclude 'label'
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:10]

print("\nRank  Feature                      Importance")
print("-" * 50)
for i, idx in enumerate(indices, 1):
    print(f"{i:2d}.   {feature_names[idx]:25s}  {importances[idx]:.4f}")

print("\n" + "=" * 70)
print("✅ ANALYSIS COMPLETE")
print("=" * 70)
