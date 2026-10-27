"""
AI Audio Detector Core
Reusable detection logic for both upload and live monitoring modes.
"""

import numpy as np
import librosa
import joblib
import os
from collections import deque
from typing import Tuple, Dict, Optional

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "audio_detection.pkl")
SAMPLE_RATE = 16000
CHUNK_DURATION = 2
DECISION_THRESHOLD = 0.40

# Global model instance
_model = None


def load_model():
    """Load the trained Random Forest model (singleton pattern)."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    return _model


def extract_features(y: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """
    Extract 61 audio features from audio signal.
    
    Args:
        y: Audio signal (numpy array)
        sr: Sample rate
        
    Returns:
        Feature vector (61 dimensions) or None if extraction fails
    """
    try:
        features = []
        
        # MFCC (13 mean + 13 std = 26 features)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        # Delta (13 std features)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features.extend(np.std(delta, axis=1))
        features.extend(np.std(delta2, axis=1))

        # Spectral features (11 features)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        features.append(np.mean(centroid))
        features.append(np.std(centroid))
        features.append(np.mean(bandwidth))
        features.append(np.std(bandwidth))
        features.extend(np.mean(contrast, axis=1))

        # Harmonic ratio (1 feature)
        y_harm, y_perc = librosa.effects.hpss(y)
        harmonic_ratio = np.mean(np.abs(y_harm)) / (np.mean(np.abs(y_perc)) + 1e-9)
        features.append(harmonic_ratio)

        return np.array(features)

    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return None


def predict_chunk(audio_chunk: np.ndarray, sr: int = SAMPLE_RATE) -> Optional[float]:
    """
    Predict AI probability for a single audio chunk.
    
    Args:
        audio_chunk: Audio signal (2 seconds recommended)
        sr: Sample rate (default 16kHz)
        
    Returns:
        AI probability (0.0 to 1.0) or None if prediction fails
    """
    try:
        model = load_model()
        features = extract_features(audio_chunk, sr)
        
        if features is None:
            return None
            
        # Reshape for sklearn
        features = features.reshape(1, -1)
        
        # Predict probability (class 1 = AI-generated)
        prob = model.predict_proba(features)[0, 1]
        
        return float(prob)
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None


def is_silent(audio_chunk: np.ndarray, threshold: float = 0.01) -> bool:
    """Check if audio chunk is silent."""
    rms = np.sqrt(np.mean(audio_chunk**2))
    return rms < threshold


def estimate_snr(audio_chunk: np.ndarray) -> float:
    """
    Estimate Signal-to-Noise Ratio (simplified).
    Returns SNR in dB (higher is better).
    """
    try:
        # Simple energy-based estimation
        signal_power = np.mean(audio_chunk**2)
        noise_estimate = np.var(audio_chunk) / 10  # Simplified
        
        if noise_estimate < 1e-10:
            return 40.0  # Very clean
            
        snr_db = 10 * np.log10(signal_power / noise_estimate)
        return float(np.clip(snr_db, -10, 40))
        
    except:
        return 20.0  # Default moderate quality


class RollingVerdict:
    """
    Rolling temporal analysis for live audio monitoring.
    Smooths predictions and detects suspicious patterns.
    """
    
    def __init__(self, 
                 history_size: int = 5,
                 threshold: float = DECISION_THRESHOLD,
                 flip_streak: int = 2):
        """
        Args:
            history_size: Number of recent predictions to track
            threshold: Decision boundary (default 0.40)
            flip_streak: Consecutive predictions needed to change state
        """
        self.history_size = history_size
        self.threshold = threshold
        self.flip_streak = flip_streak
        
        # Rolling probability history
        self.probability_history = deque(maxlen=history_size)
        
        # Session state
        self.chunk_count = 0
        self.suspicious_count = 0
        self.suspicious_streak = 0
        self.current_state = "INCONCLUSIVE"
        
        # Suspicious segments tracking
        self.in_suspicious_segment = False
        self.current_segment_start = None
        self.suspicious_segments = []
        
        # Session statistics
        self.max_probability = 0.0
        self.total_probability = 0.0
        
    def update(self, probability: float, timestamp: float = None) -> Dict:
        """
        Update rolling analysis with new prediction.
        
        Args:
            probability: AI probability for this chunk (0.0 to 1.0)
            timestamp: Optional timestamp
            
        Returns:
            Dict with current analysis state
        """
        self.chunk_count += 1
        self.probability_history.append(probability)
        
        # Update statistics
        self.total_probability += probability
        self.max_probability = max(self.max_probability, probability)
        
        # Calculate smoothed probability
        smoothed_prob = np.mean(self.probability_history) if self.probability_history else probability
        
        # Determine chunk-level classification
        is_suspicious = probability > self.threshold
        
        if is_suspicious:
            self.suspicious_count += 1
            self.suspicious_streak += 1
        else:
            self.suspicious_streak = 0
            
        # Update suspicious segments
        if is_suspicious and not self.in_suspicious_segment:
            # Start new segment
            self.in_suspicious_segment = True
            self.current_segment_start = {
                'start_chunk': self.chunk_count,
                'start_time': timestamp,
                'probabilities': [probability],
                'max_prob': probability
            }
        elif is_suspicious and self.in_suspicious_segment:
            # Continue segment
            self.current_segment_start['probabilities'].append(probability)
            self.current_segment_start['max_prob'] = max(
                self.current_segment_start['max_prob'], 
                probability
            )
        elif not is_suspicious and self.in_suspicious_segment:
            # End segment
            if len(self.current_segment_start['probabilities']) >= 2:
                # Only record segments with 2+ consecutive suspicious chunks
                self.suspicious_segments.append({
                    'start_chunk': self.current_segment_start['start_chunk'],
                    'end_chunk': self.chunk_count - 1,
                    'duration': self.chunk_count - self.current_segment_start['start_chunk'],
                    'chunk_count': len(self.current_segment_start['probabilities']),
                    'avg_probability': np.mean(self.current_segment_start['probabilities']),
                    'max_probability': self.current_segment_start['max_prob']
                })
            self.in_suspicious_segment = False
            self.current_segment_start = None
            
        # Determine overall session state
        if self.chunk_count < 3:
            # Need more data
            self.current_state = "INCONCLUSIVE"
        elif smoothed_prob > self.threshold and self.suspicious_streak >= self.flip_streak:
            self.current_state = "SUSPICIOUS AI CHARACTERISTICS"
        elif smoothed_prob <= self.threshold - 0.10:
            self.current_state = "LIKELY HUMAN"
        else:
            # In transition zone
            self.current_state = "INCONCLUSIVE"
            
        # Calculate session risk score (0-100)
        risk_score = self._calculate_risk_score(smoothed_prob)
        
        return {
            'chunk_number': self.chunk_count,
            'raw_probability': float(probability),
            'smoothed_probability': float(smoothed_prob),
            'is_suspicious': is_suspicious,
            'current_state': self.current_state,
            'suspicious_count': self.suspicious_count,
            'suspicious_streak': self.suspicious_streak,
            'avg_probability': float(self.total_probability / self.chunk_count),
            'max_probability': float(self.max_probability),
            'risk_score': risk_score,
            'suspicious_segments': len(self.suspicious_segments)
        }
        
    def _calculate_risk_score(self, smoothed_prob: float) -> int:
        """Calculate session risk score (0-100)."""
        # Factors:
        # - Smoothed probability (60% weight)
        # - Suspicious segment count (20% weight)
        # - Suspicious chunk ratio (20% weight)
        
        prob_score = smoothed_prob * 60
        
        segment_score = min(len(self.suspicious_segments) * 5, 20)
        
        suspicious_ratio = self.suspicious_count / max(self.chunk_count, 1)
        ratio_score = suspicious_ratio * 20
        
        risk = int(prob_score + segment_score + ratio_score)
        return min(max(risk, 0), 100)
        
    def get_session_summary(self) -> Dict:
        """Get complete session summary."""
        avg_prob = self.total_probability / max(self.chunk_count, 1)
        
        return {
            'total_chunks': self.chunk_count,
            'suspicious_chunks': self.suspicious_count,
            'suspicious_ratio': self.suspicious_count / max(self.chunk_count, 1),
            'average_probability': float(avg_prob),
            'max_probability': float(self.max_probability),
            'current_state': self.current_state,
            'risk_score': self._calculate_risk_score(avg_prob),
            'suspicious_segments': self.suspicious_segments,
            'segment_count': len(self.suspicious_segments)
        }
        
    def reset(self):
        """Reset session state."""
        self.probability_history.clear()
        self.chunk_count = 0
        self.suspicious_count = 0
        self.suspicious_streak = 0
        self.current_state = "INCONCLUSIVE"
        self.in_suspicious_segment = False
        self.current_segment_start = None
        self.suspicious_segments = []
        self.max_probability = 0.0
        self.total_probability = 0.0


# Module initialization
print("🎵 detector_core.py loaded")
