import os
import numpy as np
import librosa
import csv
from multiprocessing import Pool, cpu_count
from datetime import datetime
import sys

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
        return None


def process_file(args):
    path, label = args
    feats = feature_extraction(path)
    if feats is not None:
        return feats + [label]
    return None

def load_dataset(base_path, out_csv, dataset_name="Dataset"):
    tasks = []
    for label_name in ["real", "fake"]:
        folder = os.path.join(base_path, label_name)
        label = 0 if label_name == 'real' else 1
        files = [f for f in os.listdir(folder) if f.lower().endswith((".wav", ".mp3"))]
        for file in files:
            tasks.append((os.path.join(folder, file), label))
    
    total = len(tasks)
    print(f"\n{'='*70}")
    print(f"🎵 {dataset_name} - Processing {total:,} audio files")
    print(f"{'='*70}")
    print(f"⚙️  Using {cpu_count()} CPU cores")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    sys.stdout.flush()

    with Pool(cpu_count()) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(process_file, tasks), 1):
            results.append(result)
            if i % 500 == 0 or i == total:
                progress = (i / total) * 100
                bar_length = 40
                filled = int(bar_length * i / total)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"\r[{bar}] {progress:.1f}% ({i:,}/{total:,} files)", end='', flush=True)
        print()  # New line after progress

    valid_results = [r for r in results if r is not None]
    print(f"\n✅ Successfully processed: {len(valid_results):,}/{total:,} files")
    print(f"❌ Failed: {total - len(valid_results):,} files")
    
    print(f"\n💾 Writing to {out_csv}...")
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in valid_results:
            writer.writerow(row)
    
    print(f"✅ Saved {len(valid_results):,} samples to {out_csv}")
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

if __name__ == "__main__":
    BASE = os.path.join(os.path.dirname(__file__), "dataset", "for-norm", "for-norm")
    
    start_time = datetime.now()
    print("\n" + "="*70)
    print("🚀 AI AUDIO DETECTOR - FULL DATASET FEATURE EXTRACTION")
    print("="*70)
    print(f"⏰ Overall start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Training set
    load_dataset(
        os.path.join(BASE, "training"), 
        os.path.join(os.path.dirname(__file__), "train-norm.csv"),
        "TRAINING SET"
    )
    
    # Validation set
    load_dataset(
        os.path.join(BASE, "validation"), 
        os.path.join(os.path.dirname(__file__), "val-norm.csv"),
        "VALIDATION SET"
    )
    
    # Testing set
    load_dataset(
        os.path.join(BASE, "testing"), 
        os.path.join(os.path.dirname(__file__), "test-norm.csv"),
        "TESTING SET"
    )
    
    end_time = datetime.now()
    duration = end_time - start_time
    hours = int(duration.total_seconds() // 3600)
    minutes = int((duration.total_seconds() % 3600) // 60)
    
    print("\n" + "="*70)
    print("🎉 ALL FEATURE EXTRACTION COMPLETED!")
    print("="*70)
    print(f"⏰ Total time: {hours}h {minutes}m")
    print(f"✅ Ready for model training!")
    print("="*70 + "\n")
