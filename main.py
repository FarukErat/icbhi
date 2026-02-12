"""
Cosine Similarity Calculator for Healthy Respiratory Sounds
Compares healthy sounds from patients dataset vs ICBHI dataset

Features:
- Fixed-length windowing: Splits divergent-length segments into consistent windows
- Data augmentation: Augments patient segments (pitch shift, time stretch, noise)
- Overlap sliding windows: Extracts more samples from longer segments
"""

import os
import glob
import numpy as np
import pandas as pd
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Configuration
PATIENTS_DIR = "patients"
PATIENTS_DIAGNOSES_DIR = "patients/diagnoses"
ICBHI_DIR = "ICBHI_final_database"
SAMPLE_RATE = 22050  # Standard sample rate for audio processing

# Windowing and augmentation configuration
WINDOW_LENGTH = 2.0  # Fixed window length in seconds (matches ICBHI median)
WINDOW_OVERLAP = 0.5  # 50% overlap for sliding windows
MIN_SEGMENT_LENGTH = 0.5  # Minimum segment length to process (seconds)

# Augmentation settings for patients dataset
ENABLE_AUGMENTATION = True
AUGMENTATION_MULTIPLIER = 5  # Number of augmented versions per original segment


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
    Split an audio segment into fixed-length overlapping windows.

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


def augment_audio(audio, sr, num_augmentations=AUGMENTATION_MULTIPLIER):
    """
    Apply data augmentation to an audio segment.

    Augmentation techniques:
    1. Pitch shifting
    2. Time stretching
    3. Adding noise
    4. Volume variation

    Returns:
        List of augmented audio segments
    """
    augmented = []

    for i in range(num_augmentations):
        aug_audio = audio.copy()

        # Randomly select augmentation type
        aug_type = i % 5

        if aug_type == 0:
            # Pitch shift up
            aug_audio = librosa.effects.pitch_shift(aug_audio, sr=sr, n_steps=np.random.uniform(1, 3))
        elif aug_type == 1:
            # Pitch shift down
            aug_audio = librosa.effects.pitch_shift(aug_audio, sr=sr, n_steps=np.random.uniform(-3, -1))
        elif aug_type == 2:
            # Time stretch (speed up slightly)
            stretch_rate = np.random.uniform(0.9, 0.95)
            aug_audio = librosa.effects.time_stretch(aug_audio, rate=stretch_rate)
            # Resample to original length
            if len(aug_audio) > len(audio):
                aug_audio = aug_audio[:len(audio)]
            else:
                aug_audio = np.pad(aug_audio, (0, len(audio) - len(aug_audio)))
        elif aug_type == 3:
            # Time stretch (slow down slightly)
            stretch_rate = np.random.uniform(1.05, 1.1)
            aug_audio = librosa.effects.time_stretch(aug_audio, rate=stretch_rate)
            # Resample to original length
            if len(aug_audio) > len(audio):
                aug_audio = aug_audio[:len(audio)]
            else:
                aug_audio = np.pad(aug_audio, (0, len(audio) - len(aug_audio)))
        elif aug_type == 4:
            # Add white noise
            noise_level = np.random.uniform(0.001, 0.005)
            noise = np.random.randn(len(aug_audio)) * noise_level
            aug_audio = aug_audio + noise

        # Random volume adjustment
        volume_factor = np.random.uniform(0.8, 1.2)
        aug_audio = aug_audio * volume_factor

        augmented.append(aug_audio)

    return augmented


def load_patients_healthy_sounds():
    """
    Load healthy sound segments from the patients dataset.
    Healthy sounds are those tagged with 'normal' in the diagnosis CSV files.

    Applies:
    - Fixed-length windowing to handle divergent segment lengths
    - Data augmentation to increase sample count
    """
    healthy_segments = []
    original_count = 0
    windowed_count = 0
    augmented_count = 0

    # Get all wav files in patients directory
    wav_files = glob.glob(os.path.join(PATIENTS_DIR, "*.wav"))

    print(f"Found {len(wav_files)} patient audio files")
    print(f"Window length: {WINDOW_LENGTH}s, Overlap: {WINDOW_OVERLAP*100:.0f}%")
    print(f"Augmentation: {'Enabled' if ENABLE_AUGMENTATION else 'Disabled'} (x{AUGMENTATION_MULTIPLIER})")

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
                                    'augmented': False,
                                    'augmentation_type': 'original',
                                    'features': features
                                })

                                # Apply augmentation if enabled
                                if ENABLE_AUGMENTATION:
                                    aug_audios = augment_audio(window_audio, sr)
                                    aug_types = ['pitch_up', 'pitch_down', 'stretch_fast', 'stretch_slow', 'noise']

                                    for aug_idx, aug_audio in enumerate(aug_audios):
                                        aug_features = extract_audio_features(aug_audio, sr)
                                        if aug_features is not None:
                                            augmented_count += 1
                                            healthy_segments.append({
                                                'source': 'patients',
                                                'file': file_id,
                                                'start': orig_start + win_start,
                                                'end': orig_start + win_end,
                                                'original_start': orig_start,
                                                'original_end': orig_end,
                                                'original_duration': original_duration,
                                                'augmented': True,
                                                'augmentation_type': aug_types[aug_idx % len(aug_types)],
                                                'features': aug_features
                                            })
            except Exception as e:
                print(f"Error processing {wav_path}: {e}")

    print(f"\nPatients dataset summary:")
    print(f"  Original segments: {original_count}")
    print(f"  After windowing: {windowed_count}")
    print(f"  After augmentation: {augmented_count}")
    print(f"  Total segments: {len(healthy_segments)}")
    return healthy_segments


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
    return healthy_segments


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
            'patient_augmented': patient_seg.get('augmented', False),
            'patient_augmentation_type': patient_seg.get('augmentation_type', 'original'),
            'icbhi_file': icbhi_seg['file'],
            'icbhi_start': icbhi_seg['start'],
            'icbhi_end': icbhi_seg['end'],
            'icbhi_original_duration': icbhi_seg.get('original_duration', icbhi_seg['end'] - icbhi_seg['start'])
        }
        all_pairs.append(pair_data)

    # Print top 5 to console (original segments only)
    print(f"\nTop 5 Most Similar Pairs (original segments only):")
    original_pairs = [p for p in all_pairs if not p['patient_augmented']]
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

    return {
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
        'icbhi_segments': icbhi_segments
    }


def generate_markdown_report(stats, output_dir='reports'):
    """
    Generate a comprehensive markdown report and separate CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Count original vs augmented segments
    original_patient_count = sum(1 for s in stats['patients_segments'] if not s.get('augmented', False))
    augmented_patient_count = sum(1 for s in stats['patients_segments'] if s.get('augmented', False))

    # Generate summary markdown report
    report_file = os.path.join(output_dir, 'report.md')
    with open(report_file, 'w') as f:
        f.write("# Cosine Similarity Analysis Report\n\n")
        f.write("## Healthy Sounds: Patients vs ICBHI Dataset\n\n")
        f.write("---\n\n")

        # Configuration section
        f.write("## Configuration\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        f.write(f"| Window Length | {WINDOW_LENGTH}s |\n")
        f.write(f"| Window Overlap | {WINDOW_OVERLAP*100:.0f}% |\n")
        f.write(f"| Min Segment Length | {MIN_SEGMENT_LENGTH}s |\n")
        f.write(f"| Augmentation Enabled | {'Yes' if ENABLE_AUGMENTATION else 'No'} |\n")
        f.write(f"| Augmentation Multiplier | {AUGMENTATION_MULTIPLIER}x |\n")
        f.write(f"| Sample Rate | {SAMPLE_RATE} Hz |\n\n")

        # Dataset summary
        f.write("## Dataset Summary\n\n")
        f.write("| Dataset | Original Segments | Windowed/Augmented | Total |\n")
        f.write("|---------|-------------------|-------------------|-------|\n")
        f.write(f"| Patients | {original_patient_count} (original) | {augmented_patient_count} (augmented) | {len(stats['patients_segments'])} |\n")
        f.write(f"| ICBHI | - | - | {len(stats['icbhi_segments'])} |\n\n")
        f.write(f"**Total Pairs Analyzed:** {stats['matrix'].size:,}\n\n")

        # Overall statistics
        f.write("## Overall Statistics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Mean Similarity | {stats['mean']:.4f} |\n")
        f.write(f"| Median Similarity | {stats['median']:.4f} |\n")
        f.write(f"| Standard Deviation | {stats['std']:.4f} |\n")
        f.write(f"| Minimum Similarity | {stats['min']:.4f} |\n")
        f.write(f"| Maximum Similarity | {stats['max']:.4f} |\n\n")

        # Patients segment statistics
        f.write("## Patients Segments Statistics\n\n")
        f.write("Average similarity of each patient segment to all ICBHI segments:\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Mean | {np.mean(stats['patients_avg']):.4f} |\n")
        f.write(f"| Min | {np.min(stats['patients_avg']):.4f} |\n")
        f.write(f"| Max | {np.max(stats['patients_avg']):.4f} |\n\n")

        # ICBHI segment statistics
        f.write("## ICBHI Segments Statistics\n\n")
        f.write("Average similarity of each ICBHI segment to all patient segments:\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Mean | {np.mean(stats['icbhi_avg']):.4f} |\n")
        f.write(f"| Min | {np.min(stats['icbhi_avg']):.4f} |\n")
        f.write(f"| Max | {np.max(stats['icbhi_avg']):.4f} |\n\n")

        # Similarity distribution
        f.write("## Similarity Distribution\n\n")
        f.write("| Range | Count | Percentage |\n")
        f.write("|-------|-------|------------|\n")
        for dist in stats['distribution']:
            f.write(f"| {dist['range']} | {dist['count']:,} | {dist['percentage']:.2f}% |\n")
        f.write("\n")

        # Reference to CSV files
        f.write("## Data Files\n\n")
        f.write("Detailed data is available in separate CSV files:\n\n")
        f.write("- `summary_statistics.csv` - Overall statistics\n")
        f.write("- `similarity_distribution.csv` - Distribution of similarity scores\n")
        f.write("- `patients_segments.csv` - Patient segment details with average similarities\n")
        f.write("- `icbhi_segments.csv` - ICBHI segment details with average similarities\n")
        f.write("- `all_pairs.csv` - All similarity pairs (sorted by similarity descending)\n")
        f.write("- `all_pairs_original_only.csv` - Pairs with original (non-augmented) patient segments only\n\n")

        f.write("---\n\n")
        f.write("*Report generated automatically by cosine similarity analysis tool.*\n")

    print(f"Markdown report saved to '{report_file}'")

    # Save summary statistics CSV
    summary_csv = os.path.join(output_dir, 'summary_statistics.csv')
    summary_df = pd.DataFrame({
        'Metric': ['Mean Similarity', 'Median Similarity', 'Standard Deviation',
                   'Minimum Similarity', 'Maximum Similarity',
                   'Patients Segments Count (Total)', 'Patients Segments Count (Original)',
                   'Patients Segments Count (Augmented)', 'ICBHI Segments Count', 'Total Pairs',
                   'Window Length (s)', 'Window Overlap (%)', 'Augmentation Multiplier'],
        'Value': [stats['mean'], stats['median'], stats['std'],
                  stats['min'], stats['max'],
                  len(stats['patients_segments']), original_patient_count,
                  augmented_patient_count, len(stats['icbhi_segments']),
                  stats['matrix'].size,
                  WINDOW_LENGTH, WINDOW_OVERLAP * 100, AUGMENTATION_MULTIPLIER]
    })
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary statistics saved to '{summary_csv}'")

    # Save distribution CSV
    dist_csv = os.path.join(output_dir, 'similarity_distribution.csv')
    dist_df = pd.DataFrame(stats['distribution'])
    dist_df.to_csv(dist_csv, index=False)
    print(f"Distribution saved to '{dist_csv}'")

    # Save patients segments CSV with augmentation info
    patients_csv = os.path.join(output_dir, 'patients_segments.csv')
    patients_data = []
    for i, seg in enumerate(stats['patients_segments']):
        patients_data.append({
            'file': seg['file'],
            'start': seg['start'],
            'end': seg['end'],
            'original_start': seg.get('original_start', seg['start']),
            'original_end': seg.get('original_end', seg['end']),
            'original_duration': seg.get('original_duration', seg['end'] - seg['start']),
            'augmented': seg.get('augmented', False),
            'augmentation_type': seg.get('augmentation_type', 'original'),
            'avg_similarity_to_icbhi': stats['patients_avg'][i]
        })
    patients_df = pd.DataFrame(patients_data)
    patients_df.to_csv(patients_csv, index=False)
    print(f"Patients segments saved to '{patients_csv}'")

    # Save ICBHI segments CSV with original duration info
    icbhi_csv = os.path.join(output_dir, 'icbhi_segments.csv')
    icbhi_data = []
    for i, seg in enumerate(stats['icbhi_segments']):
        icbhi_data.append({
            'file': seg['file'],
            'start': seg['start'],
            'end': seg['end'],
            'original_start': seg.get('original_start', seg['start']),
            'original_end': seg.get('original_end', seg['end']),
            'original_duration': seg.get('original_duration', seg['end'] - seg['start']),
            'avg_similarity_to_patients': stats['icbhi_avg'][i]
        })
    icbhi_df = pd.DataFrame(icbhi_data)
    icbhi_df.to_csv(icbhi_csv, index=False)
    print(f"ICBHI segments saved to '{icbhi_csv}'")

    # Save all pairs CSV
    pairs_csv = os.path.join(output_dir, 'all_pairs.csv')
    print(f"Writing all pairs to '{pairs_csv}'...")
    pairs_df = pd.DataFrame(stats['all_pairs'])
    pairs_df.insert(0, 'rank', range(1, len(pairs_df) + 1))
    pairs_df.to_csv(pairs_csv, index=False)
    print(f"All pairs saved to '{pairs_csv}'")

    # Save original-only pairs CSV (excluding augmented patient segments)
    original_pairs_csv = os.path.join(output_dir, 'all_pairs_original_only.csv')
    print(f"Writing original-only pairs to '{original_pairs_csv}'...")
    original_pairs = [p for p in stats['all_pairs'] if not p.get('patient_augmented', False)]
    original_pairs_df = pd.DataFrame(original_pairs)
    original_pairs_df.insert(0, 'rank', range(1, len(original_pairs_df) + 1))
    original_pairs_df.to_csv(original_pairs_csv, index=False)
    print(f"Original-only pairs saved to '{original_pairs_csv}'")


def main():
    """
    Main function to run the cosine similarity analysis.
    """
    print("=" * 60)
    print("Cosine Similarity Analysis: Patients vs ICBHI Healthy Sounds")
    print("=" * 60)

    # Load healthy sounds from patients dataset
    print("\n[1/3] Loading healthy sounds from patients dataset...")
    patients_segments = load_patients_healthy_sounds()

    # Load healthy sounds from ICBHI dataset
    print("\n[2/3] Loading healthy sounds from ICBHI dataset...")
    icbhi_segments = load_icbhi_healthy_sounds()

    # Calculate cosine similarity
    print("\n[3/3] Calculating cosine similarity...")
    result = calculate_cosine_similarity(patients_segments, icbhi_segments)

    if result is None:
        print("Cannot calculate similarity - missing data")
        return

    similarity_matrix, patients_segs, icbhi_segs = result

    # Analyze and report results
    stats = analyze_similarity(similarity_matrix, patients_segs, icbhi_segs)

    # Generate comprehensive markdown report and CSV files
    generate_markdown_report(stats, 'reports')

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

