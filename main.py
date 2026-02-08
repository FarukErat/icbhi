"""
Cosine Similarity Calculator for Healthy Respiratory Sounds
Compares healthy sounds from patients dataset vs ICBHI dataset
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


def load_patients_healthy_sounds():
    """
    Load healthy sound segments from the patients dataset.
    Healthy sounds are those tagged with 'normal' in the diagnosis CSV files.
    """
    healthy_segments = []

    # Get all wav files in patients directory
    wav_files = glob.glob(os.path.join(PATIENTS_DIR, "*.wav"))

    print(f"Found {len(wav_files)} patient audio files")

    for wav_path in tqdm(wav_files, desc="Processing patients", unit="file"):
        file_id = os.path.basename(wav_path).replace('.wav', '')

        # Check for diagnosis files from both doctors
        fatih_csv = os.path.join(PATIENTS_DIAGNOSES_DIR, "Fatih", f"{file_id}.csv")
        guney_csv = os.path.join(PATIENTS_DIAGNOSES_DIR, "Guney", f"{file_id}.csv")

        normal_slices = []

        # Load diagnosis from Fatih
        if os.path.exists(fatih_csv):
            try:
                df = pd.read_csv(fatih_csv)
                normal_df = df[df['diagnosis'].str.lower() == 'normal']
                for _, row in normal_df.iterrows():
                    normal_slices.append((row['start'], row['end']))
            except Exception as e:
                pass

        # Load diagnosis from Guney
        if os.path.exists(guney_csv):
            try:
                df = pd.read_csv(guney_csv)
                normal_df = df[df['diagnosis'].str.lower() == 'normal']
                for _, row in normal_df.iterrows():
                    normal_slices.append((row['start'], row['end']))
            except Exception as e:
                pass

        # Extract audio segments for normal slices
        if normal_slices:
            try:
                audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE)

                for start, end in normal_slices:
                    start_sample = int(start * sr)
                    end_sample = int(end * sr)

                    if end_sample <= len(audio):
                        segment = audio[start_sample:end_sample]
                        features = extract_audio_features(segment, sr)

                        if features is not None:
                            healthy_segments.append({
                                'source': 'patients',
                                'file': file_id,
                                'start': start,
                                'end': end,
                                'features': features
                            })
            except Exception as e:
                print(f"Error processing {wav_path}: {e}")

    print(f"Extracted {len(healthy_segments)} healthy segments from patients dataset")
    return healthy_segments


def load_icbhi_healthy_sounds():
    """
    Load healthy sound segments from the ICBHI dataset.
    Healthy sounds are those where crackles=0 and wheezes=0.
    """
    healthy_segments = []

    # Get all txt annotation files
    txt_files = glob.glob(os.path.join(ICBHI_DIR, "*.txt"))

    print(f"Found {len(txt_files)} ICBHI annotation files")

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

                for start, end in normal_slices:
                    start_sample = int(start * sr)
                    end_sample = int(end * sr)

                    if end_sample <= len(audio):
                        segment = audio[start_sample:end_sample]
                        features = extract_audio_features(segment, sr)

                        if features is not None:
                            healthy_segments.append({
                                'source': 'icbhi',
                                'file': file_id,
                                'start': start,
                                'end': end,
                                'features': features
                            })
            except Exception as e:
                print(f"Error processing {wav_path}: {e}")

    print(f"Extracted {len(healthy_segments)} healthy segments from ICBHI dataset")
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
    for flat_idx in flat_indices:
        i, j = np.unravel_index(flat_idx, similarity_matrix.shape)
        similarity = similarity_matrix[i, j]
        patient_seg = patients_segments[i]
        icbhi_seg = icbhi_segments[j]
        all_pairs.append({
            'similarity': similarity,
            'patient_file': patient_seg['file'],
            'patient_start': patient_seg['start'],
            'patient_end': patient_seg['end'],
            'icbhi_file': icbhi_seg['file'],
            'icbhi_start': icbhi_seg['start'],
            'icbhi_end': icbhi_seg['end']
        })

    # Print top 5 to console
    print(f"\nTop 5 Most Similar Pairs:")
    for idx, pair in enumerate(all_pairs[:5]):
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

    # Generate summary markdown report
    report_file = os.path.join(output_dir, 'report.md')
    with open(report_file, 'w') as f:
        f.write("# Cosine Similarity Analysis Report\n\n")
        f.write("## Healthy Sounds: Patients vs ICBHI Dataset\n\n")
        f.write("---\n\n")

        # Dataset summary
        f.write("## Dataset Summary\n\n")
        f.write(f"- **Patients Segments:** {len(stats['patients_segments'])}\n")
        f.write(f"- **ICBHI Segments:** {len(stats['icbhi_segments'])}\n")
        f.write(f"- **Total Pairs Analyzed:** {stats['matrix'].size:,}\n\n")

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
        f.write("- `all_pairs.csv` - All similarity pairs (sorted by similarity descending)\n\n")

        f.write("---\n\n")
        f.write("*Report generated automatically by cosine similarity analysis tool.*\n")

    print(f"Markdown report saved to '{report_file}'")

    # Save summary statistics CSV
    summary_csv = os.path.join(output_dir, 'summary_statistics.csv')
    summary_df = pd.DataFrame({
        'Metric': ['Mean Similarity', 'Median Similarity', 'Standard Deviation',
                   'Minimum Similarity', 'Maximum Similarity',
                   'Patients Segments Count', 'ICBHI Segments Count', 'Total Pairs'],
        'Value': [stats['mean'], stats['median'], stats['std'],
                  stats['min'], stats['max'],
                  len(stats['patients_segments']), len(stats['icbhi_segments']),
                  stats['matrix'].size]
    })
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary statistics saved to '{summary_csv}'")

    # Save distribution CSV
    dist_csv = os.path.join(output_dir, 'similarity_distribution.csv')
    dist_df = pd.DataFrame(stats['distribution'])
    dist_df.to_csv(dist_csv, index=False)
    print(f"Distribution saved to '{dist_csv}'")

    # Save patients segments CSV
    patients_csv = os.path.join(output_dir, 'patients_segments.csv')
    patients_data = []
    for i, seg in enumerate(stats['patients_segments']):
        patients_data.append({
            'file': seg['file'],
            'start': seg['start'],
            'end': seg['end'],
            'avg_similarity_to_icbhi': stats['patients_avg'][i]
        })
    patients_df = pd.DataFrame(patients_data)
    patients_df.to_csv(patients_csv, index=False)
    print(f"Patients segments saved to '{patients_csv}'")

    # Save ICBHI segments CSV
    icbhi_csv = os.path.join(output_dir, 'icbhi_segments.csv')
    icbhi_data = []
    for i, seg in enumerate(stats['icbhi_segments']):
        icbhi_data.append({
            'file': seg['file'],
            'start': seg['start'],
            'end': seg['end'],
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

