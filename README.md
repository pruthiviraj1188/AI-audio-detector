# 🎵 AI Audio Detector

A high-accuracy machine learning system that detects whether audio is AI-generated or human-recorded. Built with Python, Flask, and scikit-learn.

![Accuracy](https://img.shields.io/badge/Accuracy-99.06%25-brightgreen)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Features

- ✅ **99.06% Validation Accuracy** - Professional-grade detection
- ✅ **98.94% Test Accuracy** - Reliable real-world performance
- ✅ **Multi-format Support** - MP3, WAV, MP4, OGG, FLAC, AAC, M4A, and more
- ✅ **Video Audio Extraction** - Automatically extracts audio from video files
- ✅ **Beautiful Web Interface** - Modern UI with real-time animations
- ✅ **Technical Analysis** - Detailed feature breakdown (MFCC, spectral analysis, harmonic ratio)
- ✅ **Fast Processing** - Optimized with multiprocessing

## 🚀 Demo

![AI Audio Detector Demo](demo.png)

**Live Demo:** [Coming Soon]

## 📊 Model Performance

Trained on **69,282 audio samples** from the [Fake-or-Real Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset):

| Metric | Score |
|--------|-------|
| Validation Accuracy | **99.06%** |
| Test Accuracy | **98.94%** |
| Training Samples | 53,854 |
| Validation Samples | 10,794 |
| Test Samples | 4,634 |

## 🛠️ Tech Stack

- **Backend**: Python 3.12, Flask
- **ML**: scikit-learn, librosa, NumPy, pandas
- **Audio Processing**: librosa, moviepy
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Model**: Random Forest Classifier (600 estimators)

## 📦 Installation

### Prerequisites

- Python 3.12 or higher
- pip or uv package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/pruthiviraj1188/AI-audio-detector.git
cd AI-audio-detector
```

2. **Create virtual environment**
```bash
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -e .
```

Or using `uv`:
```bash
uv sync
```

4. **Download the dataset** (optional, for training)
```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset (requires Kaggle API credentials)
cd test
kaggle datasets download -d mohammedabdeldayem/the-fake-or-real-dataset
unzip the-fake-or-real-dataset.zip -d dataset/
```

5. **Train the model** (optional, or use pre-trained)
```bash
cd test
python feature_extraction_with_progress.py  # Takes 15-30 minutes
python train.py  # Takes 5-10 minutes
```

6. **Run the web application**
```bash
cd test
python main.py
```

Visit http://127.0.0.1:5001

## 📁 Project Structure

```
AI-audio-detector/
├── test/
│   ├── main.py                              # Flask web application
│   ├── train.py                             # Model training script
│   ├── feature_extraction.py                # Full dataset feature extraction
│   ├── feature_extraction_quick.py          # Quick subset extraction
│   ├── feature_extraction_with_progress.py  # Extraction with progress bars
│   ├── audio_detection.pkl                  # Trained model (99% accuracy)
│   ├── templates/
│   │   └── index.html                       # Web interface
│   └── uploads/                             # Temporary upload directory
├── pyproject.toml                           # Project dependencies
├── README.md                                # This file
└── .gitignore                               # Git ignore rules
```

## 🎨 Features Extracted

The model analyzes 61 audio features:

1. **MFCC (Mel-Frequency Cepstral Coefficients)** - 13 mean + 13 std
2. **Delta Features** - 13 std (first derivative)
3. **Delta-Delta Features** - 13 std (second derivative)
4. **Spectral Features**:
   - Spectral Centroid (mean, std)
   - Spectral Bandwidth (mean, std)
   - Spectral Contrast (7 mean values)
5. **Harmonic Ratio** - Tonal vs percussive energy balance

## 🔧 API Usage

### Predict Audio

**Endpoint**: `POST /predict`

**Request**:
```bash
curl -X POST http://127.0.0.1:5001/predict \
  -F "file=@audio.mp3"
```

**Response**:
```json
{
  "prediction": "AI Generated audio",
  "confidence": "87.3%",
  "technical": {
    "mfcc_variance": 102.7601,
    "spectral_centroid": 1810.69,
    "harmonic_ratio": 0.9327,
    "delta_energy": 10.7881,
    "chunks_analyzed": 5
  }
}
```

## 🧪 Model Training

### Quick Training (for testing)
Uses 2,000 samples for rapid prototyping:
```bash
python feature_extraction_quick.py
python train.py
```
**Time**: ~5 minutes  
**Accuracy**: ~96.5%

### Full Training (recommended)
Uses all 69,282 samples for maximum accuracy:
```bash
python feature_extraction_with_progress.py
python train.py
```
**Time**: ~20-30 minutes  
**Accuracy**: ~99.06%

## 📈 Performance Optimization

- **Multiprocessing**: Uses all available CPU cores
- **Chunk Processing**: Analyzes audio in 2-second segments
- **Efficient Feature Extraction**: Optimized librosa usage
- **Model Size**: 120 MB (compressed Random Forest)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 To-Do / Future Enhancements

- [ ] Deploy to cloud (Heroku/Railway/AWS)
- [ ] Add user authentication
- [ ] Batch file processing
- [ ] Real-time microphone detection
- [ ] Mobile app (React Native/Flutter)
- [ ] API rate limiting and authentication
- [ ] Deep learning models (CNN/LSTM)
- [ ] Multi-language support
- [ ] Detect specific AI voice generators

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: [Fake-or-Real Dataset by Mohammed Abdeldayem](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)
- Libraries: librosa, scikit-learn, Flask, moviepy
- Inspiration: Growing concern about AI-generated audio deepfakes

## 📧 Contact

Pruthiviraj - [@pruthiviraj1188](https://github.com/pruthiviraj1188)

Project Link: [https://github.com/pruthiviraj1188/AI-audio-detector](https://github.com/pruthiviraj1188/AI-audio-detector)

---

⭐ **If you find this project useful, please consider giving it a star!** ⭐
