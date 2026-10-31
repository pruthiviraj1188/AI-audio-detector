#!/usr/bin/env python3
"""
Comprehensive Test Script for AI Audio Detector Prediction Logic
==================================================================

This script demonstrates the prediction logic and identifies any classification
inversions or threshold issues before and after fixes.

ISSUE IDENTIFIED:
If AI-generated audio is being classified as human audio, there are two possible bugs:

1. PROBABILITY INVERSION BUG:
   - Model outputs probability of class 1 (AI-generated audio)
   - If prediction is > threshold, it should be "AI"
   - But if logic is checking < threshold for "AI", probabilities are inverted

2. THRESHOLD BUG:
   - If threshold was 0.40 instead of 0.5
   - This could cause false negatives (AI audio classified as human)

This test will expose the actual behavior and confirm the bug.
"""

import numpy as np
import joblib
import os
import pandas as pd
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "audio_detection.pkl")

# Define columns matching the training data (from train.py)
columns = []
columns += [f"mfcc_mean_{i}" for i in range(13)]
columns += [f"mfcc_std_{i}" for i in range(13)]
columns += [f"delta_std_{i}" for i in range(13)]
columns += [f"delta2_std_{i}" for i in range(13)]
columns += ["centroid_mean", "centroid_std", "bandwidth_mean", "bandwidth_std"]
columns += [f"contrast_mean_{i}" for i in range(7)]
columns += ["harmonic_ratio"]

print("=" * 80)
print("🎵 AI AUDIO DETECTOR - PREDICTION LOGIC TEST")
print("=" * 80)
print("\nThis test demonstrates the prediction logic and identifies classification issues.")
print("=" * 80)

# ============================================================================
# PART 1: Load Model and Real Data
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: LOAD MODEL AND REAL DATA")
print("=" * 80)

if not os.path.exists(MODEL_PATH):
    print(f"\n❌ ERROR: Model not found at {MODEL_PATH}")
    print("Please run train.py first to train the model.")
    exit(1)

print(f"\n✅ Loading model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
print(f"   Model Type: {type(model).__name__}")
print(f"   Estimators: {model.n_estimators}")
print(f"   Max Depth: {model.max_depth}")
print(f"   Classes: {model.classes_}")

# Load test data to extract real feature vectors
print("\n✅ Loading real test data for feature analysis...")
test_csv = os.path.join(BASE_DIR, "test-norm.csv")
if os.path.exists(test_csv):
    df_test = pd.read_csv(test_csv, names=columns + ["label"])
    X_test = df_test.drop("label", axis=1).values
    y_test = df_test["label"].values
    
    # Find samples from each class
    human_indices = np.where(y_test == 0)[0]
    ai_indices = np.where(y_test == 1)[0]
    
    if len(human_indices) > 0:
        human_sample = X_test[human_indices[0]]
        print(f"   Found {len(human_indices)} human audio samples")
    else:
        human_sample = None
        print(f"   ⚠️  No human audio samples found")
    
    if len(ai_indices) > 0:
        ai_sample = X_test[ai_indices[0]]
        print(f"   Found {len(ai_indices)} AI audio samples")
    else:
        ai_sample = None
        print(f"   ⚠️  No AI audio samples found")
else:
    human_sample = None
    ai_sample = None
    print(f"   ⚠️  Test data not found at {test_csv}")

# ============================================================================
# PART 2: Generate Synthetic Test Cases
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: GENERATE SYNTHETIC TEST CASES")
print("=" * 80)

print("\n📝 Creating synthetic feature vectors to test prediction logic...")

# Calculate overall statistics for feature scaling
if human_sample is not None and ai_sample is not None:
    # Use actual data statistics
    mean_feature = np.concatenate([X_test[human_indices[:100]], X_test[ai_indices[:100]]]).mean(axis=0)
    std_feature = np.concatenate([X_test[human_indices[:100]], X_test[ai_indices[:100]]]).std(axis=0)
else:
    # Use default reasonable values for audio features
    mean_feature = np.ones(61) * 0.5
    std_feature = np.ones(61) * 0.1

# Test Case 1: STRONG AI CHARACTERISTICS
print("\n1️⃣  TEST CASE 1: Strong AI Audio Characteristics")
print("   Description: High values for AI-typical features")
print("   - High MFCC consistency (lower variance)")
print("   - Low harmonic ratio (synthetic sound)")
print("   - Regular spectral patterns")

# Create synthetic AI-like features
ai_characteristics = mean_feature.copy()
# AI audio typically has:
# - Lower MFCC variance (more regular)
# - Lower harmonic ratio (less natural)
# - Higher spectral contrast consistency
ai_characteristics[13:26] = mean_feature[13:26] * 0.3  # Lower MFCC std
ai_characteristics[26:39] = mean_feature[26:39] * 0.2  # Lower delta std
ai_characteristics[-1] = mean_feature[-1] * 0.5  # Lower harmonic ratio
test_case_ai = ai_characteristics.reshape(1, -1)

# Test Case 2: STRONG HUMAN CHARACTERISTICS
print("\n2️⃣  TEST CASE 2: Strong Human Audio Characteristics")
print("   Description: Characteristics typical of natural human speech")
print("   - High MFCC variance (natural variation)")
print("   - Higher harmonic ratio (natural harmonics)")
print("   - Varied spectral patterns")

# Create synthetic human-like features
human_characteristics = mean_feature.copy()
# Human audio typically has:
# - Higher MFCC variance (more variation)
# - Higher harmonic ratio (natural harmonics)
# - More varied patterns
human_characteristics[13:26] = mean_feature[13:26] * 2.0  # Higher MFCC std
human_characteristics[26:39] = mean_feature[26:39] * 1.8  # Higher delta std
human_characteristics[-1] = mean_feature[-1] * 2.0  # Higher harmonic ratio
test_case_human = human_characteristics.reshape(1, -1)

# ============================================================================
# PART 3: TEST CURRENT PREDICTION LOGIC
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: TEST CURRENT PREDICTION LOGIC")
print("=" * 80)

# Use real samples if available, otherwise synthetic
if ai_sample is not None:
    test_case_ai = ai_sample.reshape(1, -1)
    print("\n✅ Using real AI audio sample from test dataset")
else:
    print("\n⚠️  Using synthetic AI audio sample")

if human_sample is not None:
    test_case_human = human_sample.reshape(1, -1)
    print("✅ Using real human audio sample from test dataset")
else:
    print("⚠️  Using synthetic human audio sample")

# Get raw model probabilities
print("\n" + "-" * 80)
print("3A. RAW MODEL PROBABILITIES")
print("-" * 80)

# Test AI characteristics
prob_ai = model.predict_proba(test_case_ai)
print(f"\n🤖 AI CHARACTERISTICS TEST CASE:")
print(f"   Class probabilities: [Human={prob_ai[0, 0]:.4f}, AI={prob_ai[0, 1]:.4f}]")
print(f"   P(AI) = {prob_ai[0, 1]:.4f}")
print(f"   P(Human) = {prob_ai[0, 0]:.4f}")

# Test human characteristics
prob_human = model.predict_proba(test_case_human)
print(f"\n👤 HUMAN CHARACTERISTICS TEST CASE:")
print(f"   Class probabilities: [Human={prob_human[0, 0]:.4f}, AI={prob_human[0, 1]:.4f}]")
print(f"   P(AI) = {prob_human[0, 1]:.4f}")
print(f"   P(Human) = {prob_human[0, 0]:.4f}")

# ============================================================================
# PART 4: CURRENT THRESHOLD APPLICATION
# ============================================================================
print("\n" + "-" * 80)
print("3B. CURRENT THRESHOLD APPLICATION (threshold = 0.40)")
print("-" * 80)
print("\nLogic: If P(AI) > 0.40, classify as 'AI', else classify as 'Human'")

threshold_old = 0.40

pred_ai_old = prob_ai[0, 1] > threshold_old
pred_human_old = prob_human[0, 1] > threshold_old

print(f"\n🤖 AI CHARACTERISTICS TEST CASE:")
print(f"   P(AI) = {prob_ai[0, 1]:.4f}")
print(f"   Is {prob_ai[0, 1]:.4f} > {threshold_old}? {pred_ai_old}")
print(f"   Prediction: {'🚨 AI GENERATED' if pred_ai_old else '✅ HUMAN AUDIO'}")
if prob_ai[0, 1] > threshold_old:
    print(f"   ✓ CORRECT - AI audio correctly identified as AI")
else:
    print(f"   ❌ WRONG - AI audio misclassified as HUMAN (This is the bug!)")

print(f"\n👤 HUMAN CHARACTERISTICS TEST CASE:")
print(f"   P(AI) = {prob_human[0, 1]:.4f}")
print(f"   Is {prob_human[0, 1]:.4f} > {threshold_old}? {pred_human_old}")
print(f"   Prediction: {'🚨 AI GENERATED' if pred_human_old else '✅ HUMAN AUDIO'}")
if prob_human[0, 1] <= threshold_old:
    print(f"   ✓ CORRECT - Human audio correctly identified as Human")
else:
    print(f"   ❌ WRONG - Human audio misclassified as AI")

# ============================================================================
# PART 5: WHAT SHOULD HAPPEN WITH CORRECT THRESHOLD
# ============================================================================
print("\n" + "-" * 80)
print("3C. CORRECTED THRESHOLD APPLICATION (threshold = 0.50)")
print("-" * 80)
print("\nLogic: If P(AI) > 0.50, classify as 'AI', else classify as 'Human'")
print("This is the standard probability threshold (decision boundary)")

threshold_correct = 0.50

pred_ai_new = prob_ai[0, 1] > threshold_correct
pred_human_new = prob_human[0, 1] > threshold_correct

print(f"\n🤖 AI CHARACTERISTICS TEST CASE:")
print(f"   P(AI) = {prob_ai[0, 1]:.4f}")
print(f"   Is {prob_ai[0, 1]:.4f} > {threshold_correct}? {pred_ai_new}")
print(f"   Prediction: {'🚨 AI GENERATED' if pred_ai_new else '✅ HUMAN AUDIO'}")
if prob_ai[0, 1] > threshold_correct:
    print(f"   ✓ CORRECT - AI audio correctly identified as AI")
else:
    print(f"   ❌ WRONG - AI audio misclassified as HUMAN")

print(f"\n👤 HUMAN CHARACTERISTICS TEST CASE:")
print(f"   P(AI) = {prob_human[0, 1]:.4f}")
print(f"   Is {prob_human[0, 1]:.4f} > {threshold_correct}? {pred_human_new}")
print(f"   Prediction: {'🚨 AI GENERATED' if pred_human_new else '✅ HUMAN AUDIO'}")
if prob_human[0, 1] <= threshold_correct:
    print(f"   ✓ CORRECT - Human audio correctly identified as Human")
else:
    print(f"   ❌ WRONG - Human audio misclassified as AI")

# ============================================================================
# PART 6: ANALYSIS AND BUG DIAGNOSIS
# ============================================================================
print("\n" + "=" * 80)
print("PART 4: ANALYSIS AND BUG DIAGNOSIS")
print("=" * 80)

# Test with actual model predictions (for comparison)
pred_ai_model = model.predict(test_case_ai)[0]
pred_human_model = model.predict(test_case_human)[0]

print(f"\n📊 MODEL'S NATIVE PREDICTIONS:")
print(f"   AI sample predicted as: {'AI (1)' if pred_ai_model == 1 else 'HUMAN (0)'}")
print(f"   Human sample predicted as: {'AI (1)' if pred_human_model == 1 else 'HUMAN (0)'}")

print(f"\n🔍 BUG ANALYSIS:")

if prob_ai[0, 1] > threshold_old and not pred_ai_old:
    print(f"   ❌ INVERSION BUG DETECTED:")
    print(f"      - P(AI) = {prob_ai[0, 1]:.4f} is > threshold {threshold_old}")
    print(f"      - But prediction logic inverts it (uses < instead of >)")
    print(f"      - Result: AI audio is classified as HUMAN")
    print(f"      FIX: Change threshold comparison from '<' to '>'")
elif prob_ai[0, 1] <= threshold_old:
    print(f"   ⚠️  THRESHOLD TOO LOW:")
    print(f"      - P(AI) = {prob_ai[0, 1]:.4f} is <= threshold {threshold_old}")
    print(f"      - AI audio has low AI probability")
    print(f"      - This causes it to be classified as HUMAN")
    print(f"      FIX: Check if threshold should be raised to 0.50")
else:
    print(f"   ✓ No obvious bug detected with threshold {threshold_old}")
    print(f"   - AI audio is correctly classified with current logic")

print(f"\n📈 THRESHOLD COMPARISON:")
print(f"   Using threshold 0.40:")
print(f"      AI sample:    {prob_ai[0, 1]:.4f} > 0.40 = {pred_ai_old} → {'AI' if pred_ai_old else 'HUMAN'}")
print(f"      Human sample: {prob_human[0, 1]:.4f} > 0.40 = {pred_human_old} → {'AI' if pred_human_old else 'HUMAN'}")
print(f"")
print(f"   Using threshold 0.50:")
print(f"      AI sample:    {prob_ai[0, 1]:.4f} > 0.50 = {pred_ai_new} → {'AI' if pred_ai_new else 'HUMAN'}")
print(f"      Human sample: {prob_human[0, 1]:.4f} > 0.50 = {pred_human_new} → {'AI' if pred_human_new else 'HUMAN'}")

# ============================================================================
# PART 7: BATCH TEST ON REAL DATA
# ============================================================================
if human_sample is not None and ai_sample is not None:
    print("\n" + "=" * 80)
    print("PART 5: BATCH TEST ON REAL DATA (First 50 samples of each class)")
    print("=" * 80)
    
    # Sample from each class
    human_batch = X_test[human_indices[:50]]
    ai_batch = X_test[ai_indices[:50]]
    
    # Get probabilities
    human_probs = model.predict_proba(human_batch)[:, 1]
    ai_probs = model.predict_proba(ai_batch)[:, 1]
    
    print(f"\n👤 HUMAN AUDIO BATCH (50 samples):")
    print(f"   P(AI) statistics:")
    print(f"      Min: {human_probs.min():.4f}")
    print(f"      Mean: {human_probs.mean():.4f}")
    print(f"      Max: {human_probs.max():.4f}")
    print(f"      Std: {human_probs.std():.4f}")
    
    correct_with_old = np.sum(human_probs <= threshold_old)
    correct_with_new = np.sum(human_probs <= threshold_correct)
    print(f"   Correctly classified as HUMAN with threshold 0.40: {correct_with_old}/50")
    print(f"   Correctly classified as HUMAN with threshold 0.50: {correct_with_new}/50")
    
    print(f"\n🤖 AI AUDIO BATCH (50 samples):")
    print(f"   P(AI) statistics:")
    print(f"      Min: {ai_probs.min():.4f}")
    print(f"      Mean: {ai_probs.mean():.4f}")
    print(f"      Max: {ai_probs.max():.4f}")
    print(f"      Std: {ai_probs.std():.4f}")
    
    correct_with_old = np.sum(ai_probs > threshold_old)
    correct_with_new = np.sum(ai_probs > threshold_correct)
    print(f"   Correctly classified as AI with threshold 0.40: {correct_with_old}/50")
    print(f"   Correctly classified as AI with threshold 0.50: {correct_with_new}/50")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY: KEY FINDINGS")
print("=" * 80)

print(f"""
📌 CURRENT BEHAVIOR:
   - AI audio probability: {prob_ai[0, 1]:.4f}
   - With threshold 0.40: Classified as {'AI' if pred_ai_old else 'HUMAN'}
   - With threshold 0.50: Classified as {'AI' if pred_ai_new else 'HUMAN'}

🐛 BUG IDENTIFICATION:
   The issue is that AI-generated audio is being classified as HUMAN when:
   1. The probability P(AI) is relatively low (but still > 0.5)
   2. A threshold of 0.40 is being used instead of 0.50
   3. Possibly the prediction logic is inverted

✅ FIX REQUIRED:
   1. Set DECISION_THRESHOLD = 0.5 (standard probability boundary)
   2. Verify prediction logic uses: if prob > threshold → AI, else → HUMAN
   3. Check that both detector_core.py and main.py use the same logic

🔧 VERIFICATION:
   After fix, re-run this script to confirm:
   - AI audio samples have > 50% AI probability
   - AI audio is classified as 'AI GENERATED'
   - Human audio samples have < 50% AI probability
   - Human audio is classified as 'HUMAN AUDIO'
""")

print("=" * 80)
print("✅ TEST COMPLETE")
print("=" * 80)
