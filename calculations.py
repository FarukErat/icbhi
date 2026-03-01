"""
Calculation and feature extraction utilities for respiratory sound analysis.
"""

import numpy as np
import librosa
import librosa.feature
from sklearn.metrics.pairwise import cosine_similarity

# Configuration constants (should match main.py)
SAMPLE_RATE = 22050
WINDOW_LENGTH = 2.0
WINDOW_OVERLAP = 0.0
MIN_SEGMENT_LENGTH = 0.5


def extract_audio_features(audio_segment, sr=SAMPLE_RATE):
    """
    Extract audio features from an audio segment for comparison.
    Uses MFCC (Mel-frequency cepstral coefficients) as the feature representation.
    """
    if len(audio_segment) < sr * 0.1:  # Skip segments shorter than 0.1 seconds
        return None

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=13)

    # Compute statistics over time
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)

    # Extract additional features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_segment, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_segment, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_segment, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio_segment))

    # Combine all features into a single vector
    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        [spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate]
    ])

    return features


def split_into_windows(audio, sr, window_length=WINDOW_LENGTH, overlap=WINDOW_OVERLAP):
    """
    Split audio into fixed-length windows with optional overlap.
    Returns a list of (window_audio, start, end) tuples.
    """
    window_size = int(window_length * sr)
    step_size = int(window_size * (1 - overlap))
    windows = []
    for start in range(0, len(audio) - window_size + 1, step_size):
        end = start + window_size
        windows.append((audio[start:end], start / sr, end / sr))
    return windows


def calculate_cosine_similarity(patients_segments, icbhi_segments):
    """
    Calculate cosine similarity between healthy sounds from patients and ICBHI datasets.
    """
    if not patients_segments or not icbhi_segments:
        print("Error: One or both datasets have no healthy segments")
        return None

    # Extract feature matrices
    patients_features = np.array([s['features'] for s in patients_segments])
    icbhi_features = np.array([s['features'] for s in icbhi_segments])

    # Calculate pairwise cosine similarity
    similarity_matrix = cosine_similarity(patients_features, icbhi_features)

    return similarity_matrix, patients_segments, icbhi_segments


def analyze_similarity(similarity_matrix, patients_segments, icbhi_segments):
    """
    Analyze and report cosine similarity results.
    """
    mean_sim = np.mean(similarity_matrix)
    median_sim = np.median(similarity_matrix)
    std_sim = np.std(similarity_matrix)
    min_sim = np.min(similarity_matrix)
    max_sim = np.max(similarity_matrix)

    # Average similarity for each patients segment
    patients_avg = np.mean(similarity_matrix, axis=1)
    icbhi_avg = np.mean(similarity_matrix, axis=0)

    # Find min and max patient segments
    min_patient_idx = np.argmin(patients_avg)
    min_patient_seg = patients_segments[min_patient_idx]
    min_patient_sim = patients_avg[min_patient_idx]

    # Find max similarity patient segment with at least 30 windows
    patient_window_counts = [seg.get('window_count', 1) for seg in patients_segments]
    valid_indices = [i for i, count in enumerate(patient_window_counts) if count >= 30]
    max_valid_seg = None
    max_valid_sim = None
    if valid_indices:
        max_valid_idx = valid_indices[np.argmax([patients_avg[i] for i in valid_indices])]
        max_valid_seg = patients_segments[max_valid_idx]
        max_valid_sim = patients_avg[max_valid_idx]

    # Find highest similarity patient segments until total windows >= 30
    sorted_indices = np.argsort(patients_avg)[::-1]
    selected_indices = []
    total_windows = 0
    for idx in sorted_indices:
        count = patients_segments[idx].get('window_count', 1)
        selected_indices.append(idx)
        total_windows += count
        if total_windows >= 30:
            break
    selected_patients = [patients_segments[i] for i in selected_indices]
    selected_avg_sim = [patients_avg[i] for i in selected_indices]
    selected_total_windows = sum([seg.get('window_count', 1) for seg in selected_patients])
    selected_max_sim = max(selected_avg_sim) if selected_avg_sim else None
    selected_max_seg = selected_patients[selected_avg_sim.index(selected_max_sim)] if selected_avg_sim else None

    # Similarity distribution
    ranges = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    distribution = []
    for low, high in ranges:
        count = np.sum((similarity_matrix >= low) & (similarity_matrix < high))
        percentage = count / similarity_matrix.size * 100
        distribution.append({
            'range': f'[{low:.2f}, {high:.2f})',
            'count': int(count),
            'percentage': percentage
        })

    # All pairs data (sorted by similarity descending)
    all_pairs = []
    for i, patient_seg in enumerate(patients_segments):
        for j, icbhi_seg in enumerate(icbhi_segments):
            similarity = similarity_matrix[i, j]
            all_pairs.append({
                'patient_file': patient_seg['file'],
                'patient_start': patient_seg['start'],
                'patient_end': patient_seg['end'],
                'icbhi_file': icbhi_seg['file'],
                'icbhi_start': icbhi_seg['start'],
                'icbhi_end': icbhi_seg['end'],
                'similarity': similarity,
                'patient_augmented': patient_seg.get('augmented', False)
            })
    all_pairs.sort(key=lambda x: x['similarity'], reverse=True)

    return {
        'mean': mean_sim,
        'median': median_sim,
        'std': std_sim,
        'min': min_sim,
        'max': max_sim,
        'patients_avg': patients_avg,
        'icbhi_avg': icbhi_avg,
        'matrix': similarity_matrix,
        'patients_segments': patients_segments,
        'icbhi_segments': icbhi_segments,
        'min_patient_seg': min_patient_seg,
        'min_patient_sim': min_patient_sim,
        'max_valid_seg': max_valid_seg,
        'max_valid_sim': max_valid_sim,
        'selected_patients': selected_patients,
        'selected_total_windows': selected_total_windows,
        'selected_max_seg': selected_max_seg,
        'selected_max_sim': selected_max_sim,
        'distribution': distribution,
        'all_pairs': all_pairs
    }


def calculate_filtered_mean_similarity(similarities, segments=None):
    """
    Calculate mean similarity after removing low outliers using IQR method.
    Only segments below the lower threshold are discarded; maximum values are retained.
    Returns filtered mean, original mean, lower threshold, filtered count, original count, and optionally the indices and segments of outliers.
    """
    similarities = np.array(similarities)
    q1 = np.percentile(similarities, 25)
    q3 = np.percentile(similarities, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    mask = similarities >= lower
    filtered = similarities[mask]
    outlier_indices = np.where(~mask)[0]
    outlier_segments = [segments[i] for i in outlier_indices] if segments is not None else None
    filtered_mean = np.mean(filtered)
    original_mean = np.mean(similarities)
    return {
        'filtered_mean': filtered_mean,
        'original_mean': original_mean,
        'lower': lower,
        'filtered_count': len(filtered),
        'original_count': len(similarities),
        'outlier_indices': outlier_indices,
        'outlier_segments': outlier_segments
    }
