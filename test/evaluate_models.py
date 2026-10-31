#!/usr/bin/env python3
"""
Evaluate both old and new models to generate comparison report
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
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

print("=" * 80)
print("📊 DUAL MODEL EVALUATION FOR COMPREHENSIVE REPORT")
print("=" * 80)

# ============================================================================
# LOAD DATASETS (Same as train_improved.py uses)
# ============================================================================
print("\n📥 Loading compatible datasets (train-norm + val-norm)...")
df_train = pd.read_csv(os.path.join(BASE_DIR, "train-norm.csv"), names=columns)
df_val = pd.read_csv(os.path.join(BASE_DIR, "val-norm.csv"), names=columns)

# Combine and split as per improved training
df_combined = pd.concat([df_train, df_val], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

X_all = df_combined.drop("label", axis=1).values
y_all = df_combined["label"].values

X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
)

print(f"✅ Dataset loaded: {len(df_combined):,} total samples")
print(f"   - Training split: {len(X_train_split):,}")
print(f"   - Test split: {len(X_test_split):,}")

# ============================================================================
# LOAD MODELS
# ============================================================================
print("\n📦 Loading models...")
old_model_path = os.path.join(BASE_DIR, "audio_detection_old.pkl")
new_model_path = os.path.join(BASE_DIR, "audio_detection.pkl")

old_model = joblib.load(old_model_path)
new_model = joblib.load(new_model_path)

print(f"✅ Old model loaded")
print(f"   - Type: {type(old_model).__name__}")
print(f"   - Estimators: {old_model.n_estimators}")
print(f"   - Max Depth: {old_model.max_depth}")

print(f"✅ New model loaded")
print(f"   - Type: {type(new_model).__name__}")
print(f"   - Estimators: {new_model.n_estimators}")
print(f"   - Max Depth: {new_model.max_depth}")

# ============================================================================
# EVALUATE OLD MODEL
# ============================================================================
print("\n" + "=" * 80)
print("🔴 EVALUATING OLD MODEL (Original Overfitted)")
print("=" * 80)

# Train and test accuracy on old model
y_train_pred_old = old_model.predict(X_train_split)
old_train_acc = accuracy_score(y_train_split, y_train_pred_old)

y_test_pred_old = old_model.predict(X_test_split)
old_test_acc = accuracy_score(y_test_split, y_test_pred_old)

old_overfit_gap = (old_train_acc - old_test_acc) * 100

print(f"\nAccuracy Metrics:")
print(f"  Training Accuracy:  {old_train_acc * 100:6.2f}%")
print(f"  Test Accuracy:      {old_test_acc * 100:6.2f}%")
print(f"  Overfitting Gap:    {old_overfit_gap:6.2f}%")

# Per-class accuracy for old model
cm_old = confusion_matrix(y_test_split, y_test_pred_old)
tn_old, fp_old, fn_old, tp_old = cm_old[0][0], cm_old[0][1], cm_old[1][0], cm_old[1][1]

old_human_acc = tn_old / (tn_old + fp_old) if (tn_old + fp_old) > 0 else 0
old_ai_acc = tp_old / (fn_old + tp_old) if (fn_old + tp_old) > 0 else 0
old_balance_diff = abs(old_human_acc - old_ai_acc)

print(f"\nPer-Class Detection (Test Set):")
print(f"  Human Detection:    {old_human_acc * 100:6.2f}%")
print(f"  AI Detection:       {old_ai_acc * 100:6.2f}%")
print(f"  Balance Difference: {old_balance_diff * 100:6.2f}%")

# ============================================================================
# EVALUATE NEW MODEL
# ============================================================================
print("\n" + "=" * 80)
print("🟢 EVALUATING NEW MODEL (Improved)")
print("=" * 80)

# Train and test accuracy on new model
y_train_pred_new = new_model.predict(X_train_split)
new_train_acc = accuracy_score(y_train_split, y_train_pred_new)

y_test_pred_new = new_model.predict(X_test_split)
new_test_acc = accuracy_score(y_test_split, y_test_pred_new)

new_overfit_gap = (new_train_acc - new_test_acc) * 100

# Cross-validation on new model
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(new_model, X_train_split, y_train_split, cv=skf, scoring='accuracy', n_jobs=-1)

print(f"\nAccuracy Metrics:")
print(f"  Training Accuracy:  {new_train_acc * 100:6.2f}%")
print(f"  Test Accuracy:      {new_test_acc * 100:6.2f}%")
print(f"  CV Mean Accuracy:   {cv_scores.mean() * 100:6.2f}% (±{cv_scores.std() * 100:.2f}%)")
print(f"  Overfitting Gap:    {new_overfit_gap:6.2f}%")

# Per-class accuracy for new model
cm_new = confusion_matrix(y_test_split, y_test_pred_new)
tn_new, fp_new, fn_new, tp_new = cm_new[0][0], cm_new[0][1], cm_new[1][0], cm_new[1][1]

new_human_acc = tn_new / (tn_new + fp_new) if (tn_new + fp_new) > 0 else 0
new_ai_acc = tp_new / (fn_new + tp_new) if (fn_new + tp_new) > 0 else 0
new_balance_diff = abs(new_human_acc - new_ai_acc)

print(f"\nPer-Class Detection (Test Set):")
print(f"  Human Detection:    {new_human_acc * 100:6.2f}%")
print(f"  AI Detection:       {new_ai_acc * 100:6.2f}%")
print(f"  Balance Difference: {new_balance_diff * 100:6.2f}%")

# ============================================================================
# SAVE RESULTS FOR REPORT
# ============================================================================
results = {
    'old_train_acc': old_train_acc,
    'old_test_acc': old_test_acc,
    'old_overfit_gap': old_overfit_gap,
    'old_human_acc': old_human_acc,
    'old_ai_acc': old_ai_acc,
    'old_balance_diff': old_balance_diff,
    'old_cm': cm_old,
    
    'new_train_acc': new_train_acc,
    'new_test_acc': new_test_acc,
    'new_overfit_gap': new_overfit_gap,
    'new_human_acc': new_human_acc,
    'new_ai_acc': new_ai_acc,
    'new_balance_diff': new_balance_diff,
    'new_cm': cm_new,
    'cv_scores': cv_scores,
    
    'old_model_params': {
        'n_estimators': old_model.n_estimators,
        'max_depth': old_model.max_depth,
        'min_samples_leaf': old_model.min_samples_leaf if hasattr(old_model, 'min_samples_leaf') else 'N/A',
    },
    'new_model_params': {
        'n_estimators': new_model.n_estimators,
        'max_depth': new_model.max_depth,
        'min_samples_leaf': new_model.min_samples_leaf if hasattr(new_model, 'min_samples_leaf') else 'N/A',
    }
}

# Save as pickle for report generation
import pickle
results_file = os.path.join(BASE_DIR, "evaluation_results.pkl")
with open(results_file, 'wb') as f:
    pickle.dump(results, f)

print(f"\n✅ Results saved to evaluation_results.pkl")

# Print summary
print("\n" + "=" * 80)
print("📊 IMPROVEMENT SUMMARY")
print("=" * 80)

print(f"\nTest Accuracy Improvement:")
print(f"  Old: {old_test_acc * 100:6.2f}%")
print(f"  New: {new_test_acc * 100:6.2f}%")
print(f"  Δ:   {(new_test_acc - old_test_acc) * 100:+6.2f}%")

print(f"\nOverfitting Gap Reduction:")
print(f"  Old: {old_overfit_gap:6.2f}%")
print(f"  New: {new_overfit_gap:6.2f}%")
print(f"  Δ:   {(old_overfit_gap - new_overfit_gap):+6.2f}%")

print(f"\nClass Balance Improvement:")
print(f"  Old: {old_balance_diff * 100:6.2f}%")
print(f"  New: {new_balance_diff * 100:6.2f}%")
print(f"  Δ:   {(old_balance_diff - new_balance_diff) * 100:+6.2f}%")

print("\n" + "=" * 80)
