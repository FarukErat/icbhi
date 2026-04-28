import hashlib

import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity

SAMPLE_RATE = 22050
N_MFCC = 13
N_FFT = 2048
N_TEMPORAL_LAGS = 50

MAX_ICBHI_DURATION   = 5.0
MAX_PATIENT_DURATION = 0.4

ALGORITHMS = {
    "mfcc":      "MFCCs + cosine similarity — compare music style/timbre",
    "chroma":    "Acoustic fingerprinting — chroma features",
    "embedding": "ML embeddings — semantic/mood similarity",
    "temporal":  "Cross-correlation proxy — detect time-shifted copies",
}


def _pad(audio: np.ndarray) -> np.ndarray:
    if len(audio) < N_FFT:
        return np.pad(audio, (0, N_FFT - len(audio)))
    return audio


def _extract_mfcc(audio: np.ndarray, sr: int) -> np.ndarray:
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT)
    centroid   = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=N_FFT)
    bandwidth  = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=N_FFT)
    rolloff    = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=N_FFT)
    zcr        = librosa.feature.zero_crossing_rate(y=audio)
    return np.concatenate([
        np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
        [np.mean(centroid), np.mean(bandwidth), np.mean(rolloff), np.mean(zcr)],
    ])  # 30-dim


def _extract_chroma(audio: np.ndarray, sr: int) -> np.ndarray:
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=N_FFT, tuning=0.0)
    return np.concatenate([np.mean(chroma, axis=1), np.std(chroma, axis=1)])  # 24-dim


def _extract_embedding(audio: np.ndarray, sr: int) -> np.ndarray:
    mel      = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=N_FFT, n_mels=64)
    mel_db   = librosa.power_to_db(mel)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=N_FFT)
    chroma   = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=N_FFT, tuning=0.0)
    tonnetz  = librosa.feature.tonnetz(chroma=chroma)
    return np.concatenate([
        np.mean(mel_db, axis=1),   np.std(mel_db, axis=1),    # 128
        np.mean(contrast, axis=1), np.std(contrast, axis=1),  # 14
        np.mean(tonnetz, axis=1),  np.std(tonnetz, axis=1),   # 12
    ])  # 154-dim


def _extract_temporal(audio: np.ndarray, sr: int) -> np.ndarray:
    """Normalized waveform autocorrelation — proxy for cross-correlation matching."""
    audio_norm = audio / (np.linalg.norm(audio) + 1e-10)
    step = max(1, len(audio_norm) // 2000)
    downsampled = audio_norm[::step]
    autocorr = np.correlate(downsampled, downsampled, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr /= autocorr[0] + 1e-10
    indices = np.linspace(1, len(autocorr) - 1, N_TEMPORAL_LAGS, dtype=int)
    return autocorr[indices]  # 50-dim


def extract_all_features(audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """Extract feature vectors for all similarity algorithms from one audio chunk."""
    audio = _pad(audio)
    return {
        "mfcc":      _extract_mfcc(audio, sr),
        "chroma":    _extract_chroma(audio, sr),
        "embedding": _extract_embedding(audio, sr),
        "temporal":  _extract_temporal(audio, sr),
    }


def md5_fingerprint(audio: np.ndarray) -> str:
    """MD5 of 16-bit quantized audio — exact/re-encoded duplicate detection."""
    return hashlib.md5((audio * 32767).astype(np.int16).tobytes()).hexdigest()


# kept for backward compatibility with audio_similarity()
def extract_features(audio: np.ndarray, sr: int) -> np.ndarray:
    return _extract_mfcc(_pad(audio), sr)


def audio_similarity(file1: str, file2: str) -> dict:
    """Cosine similarity between two audio files using MFCC-based features."""
    audio1, sr1 = librosa.load(file1, sr=SAMPLE_RATE)
    audio2, sr2 = librosa.load(file2, sr=SAMPLE_RATE)
    f1 = extract_features(audio1, sr1)
    f2 = extract_features(audio2, sr2)
    return {
        "similarity":  float(cosine_similarity(f1.reshape(1, -1), f2.reshape(1, -1))[0, 0]),
        "duration_1":  len(audio1) / sr1,
        "duration_2":  len(audio2) / sr2,
        "features_1":  f1,
        "features_2":  f2,
    }
