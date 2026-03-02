"""
Calculation functions for cosine similarity analysis of respiratory sounds.
Includes feature extraction, windowing, data loading, and similarity metrics.
"""

import os
import glob
import numpy as np
import pandas as pd
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Configuration
SAMPLE_RATE = 22050  # Standard sample rate for audio processing

# Windowing configuration
WINDOW_LENGTH = 2.0  # Fixed window length in seconds (matches ICBHI median)
WINDOW_OVERLAP = 0.0  # 0% overlap for sliding windows
MIN_SEGMENT_LENGTH = 0.5  # Minimum segment length to process (seconds)

# Dataset directories
PATIENTS_DIR = "patients"
PATIENTS_DIAGNOSES_DIR = "patients/diagnoses"
ICBHI_DIR = "ICBHI_final_database"


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
    Split an audio segment into fixed-length non-overlapping windows.

    Args:
        audio: Audio signal array
        sr: Sample rate
        window_length: Length of each window in seconds
        overlap: Overlap ratio between windows (0.0 to 1.0)

    Returns:
        List of (window_audio, window_start_time, window_end_time) tuples
    """
    window_samples = int(window_length * sr)
    hop_samples = int(window_samples * (1 - overlap))

    if len(audio) < window_samples:
        # If audio is shorter than window, pad it or return as-is
        if len(audio) >= int(MIN_SEGMENT_LENGTH * sr):
            # Pad short segments to window length
            padded = np.zeros(window_samples)
            padded[:len(audio)] = audio
            return [(padded, 0.0, window_length)]
        return []

    windows = []
    start = 0
    while start + window_samples <= len(audio):
        window = audio[start:start + window_samples]
        start_time = start / sr
        end_time = (start + window_samples) / sr
        windows.append((window, start_time, end_time))
        start += hop_samples

    return windows


def load_patients_healthy_sounds():
    """
    Load healthy sound segments from the patients dataset.
    Healthy sounds are those tagged with 'normal' in the diagnosis CSV files.

    Applies fixed-length windowing to handle divergent segment lengths.
    """
    healthy_segments = []
    original_count = 0
    windowed_count = 0

    # Get all wav files in patients directory
    wav_files = glob.glob(os.path.join(PATIENTS_DIR, "*.wav"))

    print(f"Found {len(wav_files)} patient audio files")
    print(f"Window length: {WINDOW_LENGTH}s, Overlap: {WINDOW_OVERLAP*100:.0f}%")

    for wav_path in tqdm(wav_files, desc="Processing patients", unit="file"):
        file_id = os.path.basename(wav_path).replace('.wav', '')

        # Check for diagnosis files from both doctors
        fatih_csv = os.path.join(PATIENTS_DIAGNOSES_DIR, "Fatih", f"{file_id}.csv")
        guney_csv = os.path.join(PATIENTS_DIAGNOSES_DIR, "Guney", f"{file_id}.csv")

        normal_slices = []
        seen_slices = set()

        # Load diagnosis from Fatih
        if os.path.exists(fatih_csv):
            try:
                df = pd.read_csv(fatih_csv)
                normal_df = df[df['diagnosis'].str.lower() == 'normal']
                for _, row in normal_df.iterrows():
                    slice_key = (round(row['start'], 2), round(row['end'], 2))
                    if slice_key not in seen_slices:
                        seen_slices.add(slice_key)
                        normal_slices.append((row['start'], row['end']))
            except Exception as e:
                pass

        # Load diagnosis from Guney
        if os.path.exists(guney_csv):
            try:
                df = pd.read_csv(guney_csv)
                normal_df = df[df['diagnosis'].str.lower() == 'normal']
                for _, row in normal_df.iterrows():
                    slice_key = (round(row['start'], 2), round(row['end'], 2))
                    if slice_key not in seen_slices:
                        seen_slices.add(slice_key)
                        normal_slices.append((row['start'], row['end']))
            except Exception as e:
                pass

        # Extract audio segments for normal slices
        if normal_slices:
            try:
                audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE)

                for orig_start, orig_end in normal_slices:
                    start_sample = int(orig_start * sr)
                    end_sample = int(orig_end * sr)

                    if end_sample <= len(audio):
                        segment = audio[start_sample:end_sample]
                        original_duration = orig_end - orig_start
                        original_count += 1

                        # Split segment into fixed-length windows
                        windows = split_into_windows(segment, sr)

                        for window_audio, win_start, win_end in windows:
                            features = extract_audio_features(window_audio, sr)

                            if features is not None:
                                windowed_count += 1
                                healthy_segments.append({
                                    'source': 'patients',
                                    'file': file_id,
                                    'start': orig_start + win_start,
                                    'end': orig_start + win_end,
                                    'original_start': orig_start,
                                    'original_end': orig_end,
                                    'original_duration': original_duration,
                                    'features': features
                                })
            except Exception as e:
                print(f"Error processing {wav_path}: {e}")

    print(f"\nPatients dataset summary:")
    print(f"  Original segments: {original_count}")
    print(f"  After windowing: {windowed_count}")
    print(f"  Total segments: {len(healthy_segments)}")
    return healthy_segments, original_count, windowed_count


def load_icbhi_healthy_sounds():
    """
    Load healthy sound segments from the ICBHI dataset.
    Healthy sounds are those where crackles=0 and wheezes=0.

    Applies fixed-length windowing to match patient segments.
    """
    healthy_segments = []
    original_count = 0
    windowed_count = 0

    # Get all txt annotation files
    txt_files = glob.glob(os.path.join(ICBHI_DIR, "*.txt"))

    print(f"Found {len(txt_files)} ICBHI annotation files")
    print(f"Window length: {WINDOW_LENGTH}s, Overlap: {WINDOW_OVERLAP*100:.0f}%")

    for txt_path in tqdm(txt_files, desc="Processing ICBHI", unit="file"):
        wav_path = txt_path.replace('.txt', '.wav')

        if not os.path.exists(wav_path):
            continue

        normal_slices = []

        # Parse annotation file
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        start = float(parts[0])
                        end = float(parts[1])
                        crackles = int(parts[2])
                        wheezes = int(parts[3])

                        # Healthy = no crackles and no wheezes
                        if crackles == 0 and wheezes == 0:
                            normal_slices.append((start, end))
        except Exception as e:
            continue

        # Extract audio segments for normal slices
        if normal_slices:
            try:
                audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
                file_id = os.path.basename(wav_path).replace('.wav', '')

                for orig_start, orig_end in normal_slices:
                    start_sample = int(orig_start * sr)
                    end_sample = int(orig_end * sr)

                    if end_sample <= len(audio):
                        segment = audio[start_sample:end_sample]
                        original_duration = orig_end - orig_start
                        original_count += 1

                        # Split segment into fixed-length windows
                        windows = split_into_windows(segment, sr)

                        for window_audio, win_start, win_end in windows:
                            features = extract_audio_features(window_audio, sr)

                            if features is not None:
                                windowed_count += 1
                                healthy_segments.append({
                                    'source': 'icbhi',
                                    'file': file_id,
                                    'start': orig_start + win_start,
                                    'end': orig_start + win_end,
                                    'original_start': orig_start,
                                    'original_end': orig_end,
                                    'original_duration': original_duration,
                                    'features': features
                                })
            except Exception as e:
                print(f"Error processing {wav_path}: {e}")

    print(f"\nICBHI dataset summary:")
    print(f"  Original segments: {original_count}")
    print(f"  After windowing: {windowed_count}")
    print(f"  Total segments: {len(healthy_segments)}")
    return healthy_segments, original_count, windowed_count


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

    print(f"\nPatients features shape: {patients_features.shape}")
    print(f"ICBHI features shape: {icbhi_features.shape}")

    # Calculate pairwise cosine similarity
    similarity_matrix = cosine_similarity(patients_features, icbhi_features)

    return similarity_matrix, patients_segments, icbhi_segments


def analyze_similarity(similarity_matrix, patients_segments, icbhi_segments):
    """
    Analyze and report cosine similarity results.
    """
    print("\n" + "=" * 60)
    print("COSINE SIMILARITY ANALYSIS RESULTS")
    print("=" * 60)

    # Overall statistics
    mean_sim = np.mean(similarity_matrix)
    median_sim = np.median(similarity_matrix)
    std_sim = np.std(similarity_matrix)
    min_sim = np.min(similarity_matrix)
    max_sim = np.max(similarity_matrix)

    print(f"\nOverall Statistics:")
    print(f"  Mean similarity: {mean_sim:.4f}")
    print(f"  Median similarity: {median_sim:.4f}")
    print(f"  Std deviation: {std_sim:.4f}")
    print(f"  Min similarity: {min_sim:.4f}")
    print(f"  Max similarity: {max_sim:.4f}")

    # Average similarity for each patients segment
    patients_avg = np.mean(similarity_matrix, axis=1)
    print(f"\nPatients Segments Average Similarity to ICBHI:")
    print(f"  Mean: {np.mean(patients_avg):.4f}")
    print(f"  Range: [{np.min(patients_avg):.4f}, {np.max(patients_avg):.4f}]")

    # Average similarity for each ICBHI segment
    icbhi_avg = np.mean(similarity_matrix, axis=0)
    print(f"\nICBHI Segments Average Similarity to Patients:")
    print(f"  Mean: {np.mean(icbhi_avg):.4f}")
    print(f"  Range: [{np.min(icbhi_avg):.4f}, {np.max(icbhi_avg):.4f}]")

    # Get all pairs sorted by similarity
    flat_indices = np.argsort(similarity_matrix.flatten())[::-1]
    all_pairs = []
    for flat_idx in tqdm(flat_indices, desc="Sorting pairs", unit="pair"):
        i, j = np.unravel_index(flat_idx, similarity_matrix.shape)
        similarity = similarity_matrix[i, j]
        patient_seg = patients_segments[i]
        icbhi_seg = icbhi_segments[j]

        pair_data = {
            'similarity': similarity,
            'patient_file': patient_seg['file'],
            'patient_start': patient_seg['start'],
            'patient_end': patient_seg['end'],
            'patient_original_duration': patient_seg.get('original_duration', patient_seg['end'] - patient_seg['start']),
            'icbhi_file': icbhi_seg['file'],
            'icbhi_start': icbhi_seg['start'],
            'icbhi_end': icbhi_seg['end'],
            'icbhi_original_duration': icbhi_seg.get('original_duration', icbhi_seg['end'] - icbhi_seg['start'])
        }
        all_pairs.append(pair_data)

    # Print top 5 to console (original segments only)
    print(f"\nTop 5 Most Similar Pairs (original segments only):")
    original_pairs = [p for p in all_pairs if not p.get('patient_augmented', False)]
    for idx, pair in enumerate(original_pairs[:5]):
        print(f"  {idx+1}. Similarity: {pair['similarity']:.4f}")
        print(f"     Patients: {pair['patient_file']} [{pair['patient_start']:.2f}s - {pair['patient_end']:.2f}s]")
        print(f"     ICBHI: {pair['icbhi_file']} [{pair['icbhi_start']:.2f}s - {pair['icbhi_end']:.2f}s]")

    # Similarity distribution
    ranges = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    distribution = []
    print(f"\nSimilarity Distribution:")
    for low, high in ranges:
        count = np.sum((similarity_matrix >= low) & (similarity_matrix < high))
        percentage = count / similarity_matrix.size * 100
        distribution.append({'range': f"[{low:.2f}, {high:.2f})", 'count': count, 'percentage': percentage})
        print(f"  [{low:.2f}, {high:.2f}): {count} pairs ({percentage:.2f}%)")

    # Find patient segment with least similarity
    min_patient_idx = np.argmin(patients_avg)
    min_patient_seg = patients_segments[min_patient_idx]
    min_patient_sim = patients_avg[min_patient_idx]

    # Find patient segments with at least 30 windows
    patient_window_counts = [seg.get('window_count', 1) for seg in patients_segments]
    valid_indices = [i for i, count in enumerate(patient_window_counts) if count >= 30]
    if valid_indices:
        max_valid_idx = valid_indices[np.argmax(patients_avg[valid_indices])]
        max_valid_seg = patients_segments[max_valid_idx]
        max_valid_sim = patients_avg[max_valid_idx]
    else:
        max_valid_seg = None
        max_valid_sim = None

    # Select patient segments with highest similarity until total windows >= 30
    sorted_indices = np.argsort(patients_avg)[::-1]
    selected_indices = []
    total_windows = 0
    for idx in sorted_indices:
        count = patients_segments[idx].get('window_count', 1)
        selected_indices.append(idx)
        total_windows += count
        if total_windows >= 30:
            break
    if selected_indices:
        selected_patients = [patients_segments[i] for i in selected_indices]
        selected_avg_sim = [patients_avg[i] for i in selected_indices]
        selected_total_windows = sum([seg.get('window_count', 1) for seg in selected_patients])
        selected_max_sim = max(selected_avg_sim)
        selected_max_seg = selected_patients[selected_avg_sim.index(selected_max_sim)]
    else:
        selected_patients = []
        selected_total_windows = 0
        selected_max_sim = None
        selected_max_seg = None

    # Add to stats for reporting
    stats = {
        'mean': mean_sim,
        'median': median_sim,
        'std': std_sim,
        'min': min_sim,
        'max': max_sim,
        'matrix': similarity_matrix,
        'all_pairs': all_pairs,
        'distribution': distribution,
        'patients_avg': patients_avg,
        'icbhi_avg': icbhi_avg,
        'patients_segments': patients_segments,
        'icbhi_segments': icbhi_segments,
        'min_patient_seg': min_patient_seg,
        'min_patient_sim': min_patient_sim,
        'max_valid_seg': max_valid_seg,
        'max_valid_sim': max_valid_sim,
        'selected_patients': selected_patients,
        'selected_total_windows': selected_total_windows,
        'selected_max_seg': selected_max_seg,
        'selected_max_sim': selected_max_sim
    }
    return stats


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

