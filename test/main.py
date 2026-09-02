import numpy as np
import librosa
import joblib
import moviepy as mp
import tempfile
from flask import Flask, render_template, request, jsonify
import os

BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "audio_detection.pkl"))

def feature_extraction(y, sr):
    try:
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

        return np.array(features)

    except Exception:
        print("Error in feature extraction")
        return None

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

    if file_path.endswith(".mp4"):
        file_path = audio_from_video(file_path)

    chunks, sr = split_audio_into_chunks(file_path)

    if len(chunks) == 0:
        print("Audio too short!")
        return

    all_features = []

    for chunk in chunks:
        features = feature_extraction(chunk, sr)
        if features is not None:
            all_features.append(features)

    X = np.array(all_features)

    # Predict probabilities
    probs = model.predict_proba(X)[:, 1] 
    threshold = 0.40
    preds = (probs > threshold).astype(int)

    avg_prob = np.mean(probs)

    return preds ,avg_prob



app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]

    if not file.filename.lower().endswith((".mp3", ".wav", ".mp4")):
        return jsonify({"error": "Unsupported file type. Please upload MP3, WAV, or MP4."})

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    try:
        result = predict_audio(filepath)
        if result is None:
            return jsonify({"error": "Audio too short or unreadable. Please upload a longer file."})

        preds, avg_prob = result
        return jsonify({
            "prediction": "AI Generated audio" if avg_prob > 0.40 else "Human Audio"
        })
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == "__main__":
    app.run(debug=True)

