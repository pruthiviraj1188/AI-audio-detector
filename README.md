# AI Audio Authenticity Detector

A web application that detects whether an audio file is **AI Generated** or **Human Audio** using machine learning.

## How It Works

1. User uploads an audio file (MP3, WAV, or MP4) via the web interface
2. The audio is split into 2-second chunks
3. Audio features are extracted from each chunk:
   - MFCC (Mel-Frequency Cepstral Coefficients) — mean and standard deviation
   - Delta and Delta-Delta coefficients
   - Spectral Centroid, Bandwidth, and Contrast
   - Harmonic Ratio
4. A trained Random Forest Classifier predicts the probability of each chunk being AI-generated
5. The average probability across all chunks determines the final result

## Model

- Algorithm: Random Forest Classifier (600 trees)
- Dataset: [The Fake or Real Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)
- Validation Accuracy: ~99%
- Test Accuracy: ~99%

## Project Structure

```
AI-audio-detector-project/
├── test/
│   ├── templates/
│   │   └── index.html          # Web UI
│   ├── uploads/                # Temporary upload folder
│   ├── main.py                 # Flask app and prediction logic
│   ├── feature_extraction.py   # Audio feature extraction + dataset processing
│   ├── train.py                # Model training script
│   ├── audio_from_video.py     # Extract audio from video files
│   ├── model_training.ipynb    # Original training notebook
│   └── audio_detection.pkl     # Trained model (Git LFS)
├── pyproject.toml
└── README.md
```

## Setup & Installation

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- [Git LFS](https://git-lfs.com/)

### 1. Install Git LFS
```bash
git lfs install
```

### 2. Clone the Repository
```bash
git clone https://github.com/pruthiviraj1188/AI-audio-detector.git
cd AI-audio-detector
```

### 3. Install Dependencies
```bash
uv sync
```

### 4. Run the App
```bash
cd test
uv --project .. run python main.py
```

Open your browser at `http://127.0.0.1:5000`

---

## Training the Model from Scratch

If the model file is missing or you want to retrain:

### 1. Download the Dataset
```bash
cd test
uv --project .. run kaggle datasets download -d mohammedabdeldayem/the-fake-or-real-dataset
unzip the-fake-or-real-dataset.zip -d dataset
```

Or download manually from: https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset

Extract into `test/dataset/for-norm/for-norm/` with `training/`, `validation/`, `testing/` subfolders each containing `real/` and `fake/` subfolders.

### 2. Extract Features
```bash
uv --project .. run python feature_extraction.py
```
This generates `train-norm.csv`, `val-norm.csv`, and `test-norm.csv`.

### 3. Train the Model
```bash
uv --project .. run python train.py
```
This saves `audio_detection.pkl` in the `test/` directory.

### 4. Run the App
```bash
uv --project .. run python main.py
```

---

## Supported File Formats
- `.mp3`
- `.wav`
- `.mp4` (audio is extracted automatically)

## Tech Stack
- Python 3.12
- Flask — web framework
- librosa — audio feature extraction
- scikit-learn — Random Forest model
- joblib — model serialization
- moviepy — video to audio extraction
- uv — dependency management
