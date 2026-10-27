import numpy as np
import librosa
import moviepy as mp
import tempfile
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os

# Import reusable detector core
from detector_core import load_model, extract_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model once at startup (via detector_core)
model = load_model()

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".mp4", ".ogg", ".flac", ".aac", ".m4a", ".wma", ".aiff", ".aif", ".opus", ".webm", ".3gp"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Use detector_core's feature extraction (alias for compatibility)
feature_extraction = extract_features

def audio_from_video(video_path):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_audio_path = tmp.name
    tmp.close()

    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(temp_audio_path, logger=None)
    video.close()

    return temp_audio_path


def split_audio_into_chunks(file_path, chunk_duration=2):
    y, sr = librosa.load(file_path, sr=16000, mono=True)

    chunk_samples = int(chunk_duration * sr)
    total_samples = len(y)

    chunks = []

    for start in range(0, total_samples, chunk_samples):
        end = start + chunk_samples

        if end <= total_samples:
            chunk = y[start:end]
            chunks.append(chunk)

    return chunks, sr

def predict_audio(file_path):

    temp_file = None
    if file_path.lower().endswith(".mp4"):
        temp_file = audio_from_video(file_path)
        file_path = temp_file

    chunks, sr = split_audio_into_chunks(file_path)

    if len(chunks) == 0:
        print("Audio too short!")
        return None

    all_features = []

    for chunk in chunks:
        features = feature_extraction(chunk, sr)
        if features is not None:
            all_features.append(features)

    if len(all_features) == 0:
        return None

    X = np.array(all_features)

    # Predict probabilities
    probs = model.predict_proba(X)[:, 1] 
    threshold = 0.40
    preds = (probs > threshold).astype(int)

    avg_prob = np.mean(probs)

    # Technical feature averages
    avg_features = np.mean(X, axis=0)
    mfcc_variance = float(np.mean(np.std(X[:, :13], axis=0)))
    spectral_centroid = float(avg_features[52])  # centroid_mean index
    harmonic_ratio = float(avg_features[-1])
    delta_energy = float(np.mean(avg_features[26:39]))

    if temp_file and os.path.exists(temp_file):
        os.remove(temp_file)

    return preds, avg_prob, {
        "mfcc_variance": round(mfcc_variance, 4),
        "spectral_centroid": round(spectral_centroid, 2),
        "harmonic_ratio": round(harmonic_ratio, 4),
        "delta_energy": round(delta_energy, 4),
        "chunks_analyzed": len(chunks)
    }



app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "uploads")
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported file type. Use .wav, .mp3, or .mp4"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        result = predict_audio(filepath)
        if result is None:
            return jsonify({"error": "Audio too short or could not be processed"}), 422
        preds, avg_prob, tech = result
        return jsonify({
            "prediction": "AI Generated audio" if avg_prob > 0.40 else "Human Audio",
            "confidence": f"{avg_prob * 100:.1f}%",
            "technical": tech
        })
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/static/<path:filename>")
def serve_static(filename):
    """Serve static files (JS, CSS, JSON, etc.)"""
    static_dir = os.path.join(BASE_DIR, 'static')
    from flask import send_from_directory
    
    # Set proper content type for JSON files
    if filename.endswith('.json'):
        response = send_from_directory(static_dir, filename)
        response.headers['Content-Type'] = 'application/json'
        return response
    
    return send_from_directory(static_dir, filename)


@app.route("/health")
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "1.0.0"
    })


if __name__ == "__main__":
    print("=" * 70)
    print("🎵 AI AUDIO AUTHENTICITY DETECTOR")
    print("=" * 70)
    print("📊 Model: Random Forest (99.06% accuracy)")
    print("🌐 Web App: http://127.0.0.1:5001/")
    print("   • Upload audio/video files for analysis")
    print("💊 Health Check: http://127.0.0.1:5001/health")
    print("🌍 Languages: English, Hindi, Tamil, Telugu, Malayalam")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5001)
