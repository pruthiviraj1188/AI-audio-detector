import os
import numpy as np
import librosa
import csv
from multiprocessing import Pool, cpu_count

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


def process_file(args):
    path, label = args
    feats = feature_extraction(path)
    if feats is not None:
        return feats + [label]
    return None

def load_dataset(base_path, out_csv, max_per_class=1000):
    tasks = []
    for label_name in ["real", "fake"]:
        folder = os.path.join(base_path, label_name)
        label = 0 if label_name == 'real' else 1
        files = [f for f in os.listdir(folder) if f.lower().endswith((".wav", ".mp3"))]
        # Limit to max_per_class files
        files = files[:max_per_class]
        for file in files:
            tasks.append((os.path.join(folder, file), label))
    
    print(f"Processing {len(tasks)} files...")

    with Pool(cpu_count()) as pool:
        results = pool.map(process_file, tasks)

    print(f"Writing to {out_csv}...")
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in results:
            if row is not None:
                writer.writerow(row)
    
    print(f"Done! Wrote {sum(1 for r in results if r is not None)} samples")

if __name__ == "__main__":
    BASE = os.path.join(os.path.dirname(__file__), "dataset", "for-norm", "for-norm")
    print("Extracting training features (1000 per class)...")
    load_dataset(os.path.join(BASE, "training"), os.path.join(os.path.dirname(__file__), "train-norm.csv"), max_per_class=1000)
    print("\nExtracting validation features (200 per class)...")
    load_dataset(os.path.join(BASE, "validation"), os.path.join(os.path.dirname(__file__), "val-norm.csv"), max_per_class=200)
    print("\nExtracting testing features (200 per class)...")
    load_dataset(os.path.join(BASE, "testing"), os.path.join(os.path.dirname(__file__), "test-norm.csv"), max_per_class=200)
    print("\nAll done!")
