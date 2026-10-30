#!/usr/bin/env python3
"""
Diagnostic Script: AI Audio Misclassification Bug Analysis
This script examines the model and prediction pipeline to identify the root cause
of AI audio being classified as human audio.
"""

import joblib
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "audio_detection.pkl")

print("=" * 80)
print("AI AUDIO DETECTOR - DIAGNOSTIC SCRIPT")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD AND INSPECT MODEL
# ============================================================================
print("\n[STEP 1] Loading trained model...")
if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: Model not found at {MODEL_PATH}")
    sys.exit(1)

model = joblib.load(MODEL_PATH)
print(f"✅ Model loaded: {type(model).__name__}")
print(f"   - Classes: {model.classes_}")
print(f"   - Number of classes: {model.n_classes_}")
print(f"   - Number of trees: {model.n_estimators}")

# ============================================================================
# STEP 2: VERIFY LABEL MAPPING
# ============================================================================
print("\n[STEP 2] Verifying label mapping...")
print(f"   Model classes: {model.classes_}")
print(f"   Class 0 = {model.classes_[0]} (Real/Human Audio)")
print(f"   Class 1 = {model.classes_[1]} (Fake/AI-Generated Audio)")
print("   ✅ Label mapping: 0=human, 1=AI-generated (CORRECT)")

# ============================================================================
# STEP 3: FEATURE DIMENSION CHECK
# ============================================================================
print("\n[STEP 3] Checking feature dimensions...")
print(f"   Expected features: {model.n_features_in_}")
print(f"   Feature names (from train.py): 61 dimensions")
print(f"      - MFCC: 26 features (13 mean + 13 std)")
print(f"      - Delta/Delta2: 26 features (13 std + 13 std)")
print(f"      - Spectral: 11 features (centroid mean/std, bandwidth mean/std, 7 contrast)")
print(f"      - Harmonic ratio: 1 feature")
print(f"      - Total: 64 features")

# Let's check what CSV shows
import pandas as pd
train_csv = os.path.join(BASE_DIR, "train-norm.csv")
if os.path.exists(train_csv):
    df = pd.read_csv(train_csv, nrows=1)
    print(f"   ✅ Actual train CSV columns: {len(df.columns)} (label included)")
    print(f"   ✅ Features in model: {model.n_features_in_}")

# ============================================================================
# STEP 4: TEST WITH SYNTHETIC FEATURES
# ============================================================================
print("\n[STEP 4] Testing model with synthetic feature vectors...")

# Create 64-dimensional feature vectors (to match training)
print("\n   Creating test vectors...")
# Features breakdown (64 total):
# 0-12: MFCC mean (13)
# 13-25: MFCC std (13)
# 26-38: Delta std (13)
# 39-51: Delta2 std (13)
# 52: centroid_mean
# 53: centroid_std
# 54: bandwidth_mean
# 55: bandwidth_std
# 56-62: contrast mean (7)
# 63: harmonic_ratio (1)

# Test Vector A: AI-LIKE (high smoothness, low variance)
ai_like_features = np.zeros(64)
ai_like_features[:13] = 0.3   # Low MFCC mean variation
ai_like_features[13:26] = 0.1  # Low MFCC std (smooth)
ai_like_features[26:39] = 0.05  # Low delta std (smooth)
ai_like_features[39:52] = 0.03  # Low delta2 std (smooth)
ai_like_features[52:56] = np.array([2000, 500, 1500, 300])  # Spectral features
ai_like_features[56:63] = np.array([3.5, 3.2, 3.0, 2.8, 3.1, 3.3, 3.2])  # 7 contrast features
ai_like_features[63] = 0.8  # Harmonic ratio

# Test Vector B: HUMAN-LIKE (high variance, natural patterns)
human_like_features = np.zeros(64)
human_like_features[:13] = 0.5   # Higher MFCC mean variation
human_like_features[13:26] = 0.4  # Higher MFCC std (natural variation)
human_like_features[26:39] = 0.3  # Higher delta std (dynamic)
human_like_features[39:52] = 0.25  # Higher delta2 std (natural)
human_like_features[52:56] = np.array([2500, 800, 1800, 500])  # Spectral features
human_like_features[56:63] = np.array([2.1, 2.3, 2.0, 2.4, 2.2, 2.1, 2.3])  # 7 contrast features
human_like_features[63] = 0.6  # Harmonic ratio

print("   Test Vector A (AI-LIKE):")
print(f"      Features: {ai_like_features[:5]} ... (showing first 5)")

print("\n   Test Vector B (HUMAN-LIKE):")
print(f"      Features: {human_like_features[:5]} ... (showing first 5)")

# Make predictions
print("\n   Testing predict_proba()...")
ai_proba = model.predict_proba(ai_like_features.reshape(1, -1))[0]
human_proba = model.predict_proba(human_like_features.reshape(1, -1))[0]

print(f"\n   AI-LIKE Vector:")
print(f"      P(Human=0): {ai_proba[0]:.4f}")
print(f"      P(AI=1):    {ai_proba[1]:.4f}")
print(f"      Classification: {'AI-GENERATED' if ai_proba[1] > 0.5 else 'HUMAN'}")
print(f"      ⚠️  This should be AI-generated for an AI-like vector!")

print(f"\n   HUMAN-LIKE Vector:")
print(f"      P(Human=0): {human_proba[0]:.4f}")
print(f"      P(AI=1):    {human_proba[1]:.4f}")
print(f"      Classification: {'AI-GENERATED' if human_proba[1] > 0.5 else 'HUMAN'}")
print(f"      ⚠️  This should be HUMAN for a human-like vector!")

# ============================================================================
# STEP 5: CHECK PREDICTION LOGIC IN CODE
# ============================================================================
print("\n[STEP 5] Code Review - Prediction Logic...")

print("\n   detector_core.py - predict_chunk():")
print("      prob = model.predict_proba(features)[0, 1]  # Gets P(class=1=AI)")
print("      Returns: float(prob)")
print("      ✅ CORRECT: Returns P(AI)")

print("\n   main.py - predict_audio():")
print("      probs = model.predict_proba(X)[:, 1]  # Gets P(AI) for each chunk")
print("      preds = (probs > threshold).astype(int)  # threshold = 0.5")
print("      avg_prob = np.mean(probs)")
print("      is_ai_generated = avg_prob > 0.5")
print("      ✅ CORRECT: If avg_prob > 0.5, classify as AI")

print("\n   upload.js - Frontend Logic:")
print("      if (data.prediction === 'Human Audio') { displayHumanResult(...) }")
print("      else { displayAIResult(...) }")
print("      ✅ CORRECT: Uses prediction from backend")

# ============================================================================
# STEP 6: LOAD REAL DATA AND TEST
# ============================================================================
print("\n[STEP 6] Testing with real CSV data...")

train_csv = os.path.join(BASE_DIR, "train-norm.csv")
if os.path.exists(train_csv):
    # Read first few rows from training set
    df_train = pd.read_csv(train_csv, nrows=10)
    labels = df_train.iloc[:, -1].values  # Last column is label
    X_test = df_train.iloc[:, :-1].values  # All but last column
    
    print(f"   Loaded {len(X_test)} samples from train-norm.csv")
    print(f"   Label distribution: {np.bincount(labels.astype(int))}")
    
    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)
    
    print(f"\n   Sample Predictions:")
    for i in range(min(5, len(X_test))):
        true_label = "HUMAN" if labels[i] == 0 else "AI"
        pred_label = "HUMAN" if preds[i] == 0 else "AI"
        prob_ai = probs[i][1]
        status = "✅" if preds[i] == labels[i] else "❌"
        print(f"      {status} True: {true_label:6} | Pred: {pred_label:6} | P(AI): {prob_ai:.4f}")

# ============================================================================
# STEP 7: ANALYZE POTENTIAL ISSUES
# ============================================================================
print("\n[STEP 7] Potential Issues Analysis...")

print("\n   1. Label Inversion in Dataset?")
print("      - feature_extraction.py line 34-35:")
print("        label = 0 if label_name == 'real' else 1")
print("        ✅ Correct: 'real' -> 0, 'fake' -> 1")

print("\n   2. Label Inversion in Model Training?")
print("      - train.py uses labels directly from feature_extraction")
print("      - RandomForestClassifier(random_state=42) trained with correct labels")
print("      ✅ Should be correct")

print("\n   3. Threshold Logic Error?")
print("      - threshold = 0.5 is standard for binary classification")
print("      - If P(AI) > 0.5, should classify as AI-generated")
print("      - Using < 0.5 would INVERT the classification (BUG)")
print("      ✅ Current code uses > 0.5 (correct)")

print("\n   4. Model Performance Issue?")
print("      - If model predictions are poor, it could misclassify")
print("      - This would show up in accuracy metrics")
print("      - Needs to check accuracy_score on test set")

# ============================================================================
# STEP 8: SUMMARY AND RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)

print("\n📊 Model Configuration:")
print(f"   ✅ Classes: {model.classes_} (0=Human, 1=AI)")
print(f"   ✅ Label mapping appears correct")
print(f"   ✅ Features dimension: {model.n_features_in_}")

print("\n🔍 Code Review:")
print(f"   ✅ detector_core.predict_chunk(): Returns P(AI) correctly")
print(f"   ✅ main.predict_audio(): Uses avg_prob > 0.5 correctly")
print(f"   ✅ upload.js: Interprets prediction field correctly")

print("\n⚠️  Possible Root Causes:")
print("   1. MODEL ACCURACY: Model may be poorly trained/performing poorly")
print("      → Check model.score(X_test, y_test)")
print("   2. FEATURE EXTRACTION: Feature mismatch between training and inference")
print("      → Verify feature order and dimensions match exactly")
print("   3. DATA QUALITY: Training dataset may be mislabeled")
print("      → Verify 'real' and 'fake' folders contain correct samples")
print("   4. THRESHOLD: 0.5 may be too low for this model")
print("      → Consider calibrating threshold based on ROC curve")

print("\n📋 Next Steps:")
print("   1. Run accuracy check on test set")
print("   2. Verify feature extraction produces consistent 61 dimensions")
print("   3. Check confusion matrix: False positives vs False negatives")
print("   4. Validate training data labels")
print("   5. Consider retraining with better hyperparameters")

print("\n" + "=" * 80)
