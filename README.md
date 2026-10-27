# 🎵 AI Audio Detector

A professional-grade machine learning system that detects whether audio is AI-generated or human-recorded with **99% accuracy**. Upload audio files for instant analysis with comprehensive technical breakdown.

![Accuracy](https://img.shields.io/badge/Accuracy-99.06%25-brightgreen)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Features

### 📤 **Web App - File Upload**
- Upload audio/video files (MP3, WAV, MP4, OGG, FLAC, AAC, M4A, etc.)
- Instant AI detection with 99% accuracy
- Technical analysis breakdown (MFCC, spectral features, harmonic ratio)
- Multi-format support with automatic video audio extraction
- Beautiful UI with 5 language support

### 🔬 **Core Capabilities**
- ✅ **99.06% Validation Accuracy** - Professional-grade detection
- ✅ **98.94% Test Accuracy** - Reliable real-world performance
- ✅ **Multi-format Support** - Supports 12+ audio/video formats
- ✅ **Video Audio Extraction** - Automatically extracts audio from videos
- ✅ **Multi-Language Interface** - English, Hindi, Tamil, Telugu, Malayalam
- ✅ **Technical Analysis** - Detailed feature breakdown with explanations
- ✅ **Fast Processing** - Optimized with multiprocessing

## 🚀 Quick Start

1. **Start the server:**
```bash
cd test
python main.py
```

2. **Open in browser:** http://127.0.0.1:5001/

3. **Upload & Analyze:**
   - Drag & drop audio/video file
   - Click "Analyze Audio"
   - View results with technical details

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

**Backend:**
- Python 3.12, Flask
- scikit-learn, librosa, NumPy, pandas

**ML Model:**
- Random Forest Classifier (600 estimators)
- 61 audio features (MFCC, spectral, harmonic)
- Trained on 69,282 samples

**Web App:**
- HTML5, CSS3, JavaScript (Vanilla)
- Multi-language support (i18n)
- Responsive design with animations

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

6. **Run the application**
```bash
cd test
python main.py
```

Access the application:
- **Web App**: http://127.0.0.1:5001/
- **Health Check**: http://127.0.0.1:5001/health

## 📁 Project Structure

```
AI-audio-detector/
├── test/
│   ├── main.py                              # Flask server
│   ├── detector_core.py                     # ML detection logic & features
│   ├── train.py                             # Model training script
│   ├── feature_extraction_with_progress.py  # Feature extraction with progress
│   ├── audio_detection.pkl                  # Trained model (99% accuracy)
│   ├── templates/
│   │   └── index.html                       # Main web interface
│   ├── static/
│   │   ├── translations.json                # 5-language translations
│   │   ├── i18n.js                          # Internationalization logic
│   │   └── upload.js                        # Upload mode logic
│   └── uploads/                             # Temporary upload directory
├── pyproject.toml                           # Project dependencies
├── requirements.txt                         # Pip dependencies
├── README.md                                # This file
├── LICENSE                                  # MIT License
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

## 🔧 Usage

### Web App - Upload Mode

**Web Interface**: http://127.0.0.1:5001/
1. Drag & drop or click to upload audio/video file
2. Supported formats: MP3, WAV, MP4, OGG, FLAC, AAC, M4A, etc.
3. Click "Analyze Audio"
4. View results with:
   - AI/Human prediction
   - Confidence percentage
   - Technical feature breakdown
   - Reasoning explanation

**API Endpoint**: `POST /predict`

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

## ⚠️ **Important Limitations & Considerations**

### Model Performance Considerations

**Optimal Conditions**:
- Clean audio (minimal background noise)
- 16kHz sample rate
- 2-second or longer segments
- Uncompressed or lightly compressed audio

**Performance May Decrease With**:
- Heavy compression (low bitrate MP3)
- Background noise
- Echo, reverb, or acoustic distortion
- Microphone quality variations
- Unseen AI voice generators
- Audio processing (denoise, normalize, etc.)

### Scientific Honesty

⚠️ **This system provides probabilistic assessments, not proof**
- Suspicious results ≠ definitive proof of AI generation
- Results should be reviewed by humans
- Use as decision-support, not final verdict
- Model trained on specific dataset (may not generalize to all AI generators)

## 🌐 **Use Cases**

This system is domain-agnostic and applicable to:

1. **Content Verification** - YouTube videos, podcasts, social media
2. **Media Authenticity** - Journalism and news verification
3. **Education** - Academic integrity monitoring
4. **HR & Recruitment** - Interview authenticity verification
5. **Customer Support** - Call quality monitoring
6. **Fraud Prevention** - Deepfake and social engineering detection
7. **Research** - AI voice generation analysis
8. **Personal Use** - Verify audio sources you encounter online

## 📋 To-Do / Future Enhancements

### Completed ✅
- [x] Upload mode with 99% accuracy
- [x] Multi-language interface (5 languages)
- [x] Technical feature analysis
- [x] Multi-format support with video extraction

### Planned 🚧
- [ ] Desktop app for system audio
- [ ] Deploy to cloud (Heroku/Railway/AWS)
- [ ] User authentication and history
- [ ] Batch file processing
- [ ] Report export (PDF/CSV)
- [ ] API rate limiting and authentication
- [ ] Deep learning models (CNN/LSTM)
- [ ] Detect specific AI voice generators
- [ ] Mobile app (React Native/Flutter)
- [ ] Webhook notifications
- [ ] Admin dashboard
- [ ] Usage analytics

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
