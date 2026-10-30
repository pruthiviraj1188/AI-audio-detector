#!/usr/bin/env python3
"""
Accuracy Check: Evaluate model performance on test data
"""

import joblib
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "audio_detection.pkl")

print("=" * 80)
print("MODEL ACCURACY CHECK")
print("=" * 80)

# Load model
model = joblib.load(MODEL_PATH)
print(f"\n✅ Model loaded: {type(model).__name__}")

# Load test data
test_csv = os.path.join(BASE_DIR, "test-norm.csv")
print(f"\n📂 Loading test data from: {test_csv}")
df_test = pd.read_csv(test_csv)
print(f"   Total samples: {len(df_test)}")

# Separate features and labels
X_test = df_test.iloc[:, :-1].values
y_test = df_test.iloc[:, -1].values

print(f"   Feature dimensions: {X_test.shape}")
print(f"   Label distribution: {np.bincount(y_test.astype(int))}")
print(f"   Human (0): {np.sum(y_test == 0)} samples")
print(f"   AI (1): {np.sum(y_test == 1)} samples")

# Get predictions
print(f"\n🔮 Making predictions...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Calculate accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\n📊 Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\n📋 Confusion Matrix:")
print(f"   True Negatives (TN):  {tn}  - Correctly identified HUMAN")
print(f"   False Positives (FP): {fp}  - Human marked as AI")
print(f"   False Negatives (FN): {fn}  - AI marked as HUMAN ⚠️")
print(f"   True Positives (TP):  {tp}  - Correctly identified AI")

# Classification metrics
print(f"\n📈 Classification Metrics:")
print(f"   Precision (AI): {tp / (tp + fp):.4f}")  # Of predicted AI, how many were correct
print(f"   Recall (AI):    {tp / (tp + fn):.4f}")  # Of actual AI, how many were found
print(f"   F1 Score:       {2 * tp / (2*tp + fp + fn):.4f}")

# Probability analysis
print(f"\n🎯 Probability Analysis (P(AI)):")
p_ai = y_pred_proba[:, 1]
print(f"   Min P(AI):     {np.min(p_ai):.4f}")
print(f"   Max P(AI):     {np.max(p_ai):.4f}")
print(f"   Mean P(AI):    {np.mean(p_ai):.4f}")
print(f"   Median P(AI):  {np.median(p_ai):.4f}")
print(f"   Std P(AI):     {np.std(p_ai):.4f}")

# Count predictions at different thresholds
print(f"\n🔢 Predictions by Threshold:")
for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    preds_at_thresh = (p_ai > thresh).sum()
    print(f"   Threshold {thresh}: {preds_at_thresh} AI predictions ({100*preds_at_thresh/len(p_ai):.1f}%)")

# Detailed classification report
print(f"\n📑 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))

# ROC Analysis
fpr, tpr, thresholds = roc_curve(y_test, p_ai)
roc_auc = auc(fpr, tpr)
print(f"\n📊 ROC-AUC Score: {roc_auc:.4f}")

# Find optimal threshold
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"   Optimal threshold (Youden): {optimal_threshold:.4f}")

# Verify dataset labels match expected format
print(f"\n✅ Dataset Validation:")
print(f"   Unique labels in test set: {np.unique(y_test)}")
print(f"   Expected: [0, 1] where 0=Human, 1=AI")

# Issue detection
print(f"\n⚠️  ISSUE DETECTION:")
if fn > tp and fn > 0:
    print(f"   🚨 HIGH FALSE NEGATIVE RATE: {fn} AI samples marked as HUMAN")
    print(f"      This matches the reported bug!")
if np.mean(p_ai) < 0.3:
    print(f"   🚨 MODEL BIAS: Average P(AI) is very low ({np.mean(p_ai):.4f})")
    print(f"      Model tends to predict HUMAN regardless of input")

print("\n" + "=" * 80)
