import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity

SAMPLE_RATE = 22050
N_MFCC = 13

MAX_ICBHI_DURATION    = 5.0   # ICBHI healthy segments: at most 5 seconds
MAX_PATIENT_DURATION  = 0.4   # Patient ill segments: at most 0.4 seconds


N_FFT = 2048


def extract_features(audio: np.ndarray, sr: int) -> np.ndarray:
    if len(audio) < N_FFT:
        audio = np.pad(audio, (0, N_FFT - len(audio)))
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=N_FFT)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=N_FFT)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=N_FFT)
    zcr = librosa.feature.zero_crossing_rate(y=audio)

    return np.concatenate([
        np.mean(mfccs, axis=1),
        np.std(mfccs, axis=1),
        [
            np.mean(spectral_centroid),
            np.mean(spectral_bandwidth),
            np.mean(spectral_rolloff),
            np.mean(zcr),
        ],
    ])


def audio_similarity(file1: str, file2: str) -> dict:
    """
    Compute cosine similarity between two audio files using MFCC-based features.

    Returns a dict with:
      - similarity: float in [-1, 1], higher means more similar
      - duration_1, duration_2: durations of each file in seconds
      - features_1, features_2: raw feature vectors
    """
    audio1, sr1 = librosa.load(file1, sr=SAMPLE_RATE)
    audio2, sr2 = librosa.load(file2, sr=SAMPLE_RATE)

    features1 = extract_features(audio1, sr1)
    features2 = extract_features(audio2, sr2)

    similarity = float(cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))[0, 0])

    return {
        "similarity": similarity,
        "duration_1": len(audio1) / sr1,
        "duration_2": len(audio2) / sr2,
        "features_1": features1,
        "features_2": features2,
    }
