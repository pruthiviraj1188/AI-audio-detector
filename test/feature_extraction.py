import os
import numpy as np
import pandas as pd
import librosa
import csv

def feature_extraction(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000, mono=True)

        features = []
    
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        # Delta
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        features.extend(np.std(delta, axis=1))
        features.extend(np.std(delta2, axis=1))

        # Spectral features
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        features.append(np.mean(centroid))
        features.append(np.std(centroid))
        features.append(np.mean(bandwidth))
        features.append(np.std(bandwidth))
        features.extend(np.mean(contrast, axis=1))

        # Harmonic ratio
        y_harm, y_perc = librosa.effects.hpss(y)
        harmonic_ratio = np.mean(np.abs(y_harm)) / (np.mean(np.abs(y_perc)) + 1e-9)
        features.append(harmonic_ratio)

        return features

    except Exception as e:
        print(f"error in feature extraction: {e}")
        return None


def load_dataset(base_path, out_csv):
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)

        for label_name in ["real", "fake"]:
            folder = os.path.join(base_path, label_name)

            for file in os.listdir(folder):
                if file.lower().endswith((".wav", ".mp3")):
                    path = os.path.join(folder, file)
                    feats = feature_extraction(path)

                    if feats is not None:
                        label = 0 if label_name == 'real' else 1
                        writer.writerow(feats + [label])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to dataset root (contains training/, validation/, testing/)")
    args = parser.parse_args()

    load_dataset(os.path.join(args.dataset, "training"), "train-norm.csv")
    load_dataset(os.path.join(args.dataset, "validation"), "val-norm.csv")
    load_dataset(os.path.join(args.dataset, "testing"), "test-norm.csv")
