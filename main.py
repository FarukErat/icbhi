"""
Cosine Similarity Calculator for Healthy Respiratory Sounds
Compares healthy sounds from patients dataset vs ICBHI dataset

Features:
- Fixed-length windowing: Splits divergent-length segments into consistent windows
- Overlap sliding windows: Extracts more samples from longer segments
"""

import os
import numpy as np
import pandas as pd
import warnings

import calculations

warnings.filterwarnings('ignore')



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

        # Configuration section
        f.write("## Configuration\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        f.write(f"| Window Length | {calculations.WINDOW_LENGTH}s |\n")
        f.write(f"| Window Overlap | {calculations.WINDOW_OVERLAP*100:.0f}% |\n")
        f.write(f"| Min Segment Length | {calculations.MIN_SEGMENT_LENGTH}s |\n")
        f.write(f"| Sample Rate | {calculations.SAMPLE_RATE} Hz |\n\n")

        # Dataset summary
        f.write("## Dataset Summary\n\n")
        f.write("| Dataset | Original Segments | Windowed Segments | Total |\n")
        f.write("|---------|-------------------|-------------------|-------|\n")
        f.write(f"| Patients | {stats['patients_original_count']} | {stats['patients_windowed_count']} | {len(stats['patients_segments'])} |\n")
        f.write(f"| ICBHI | {stats['icbhi_original_count']} | {stats['icbhi_windowed_count']} | {len(stats['icbhi_segments'])} |\n\n")
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

        # Add least similarity patient segment info
        f.write("## Least Similar Patient Segment\n\n")
        min_seg = stats['min_patient_seg']
        min_sim = stats['min_patient_sim']
        f.write(f"The patient segment causing the least similarity is:\n\n")
        f.write(f"- File: {min_seg['file']}\n")
        f.write(f"- Start: {min_seg['start']}s\n")
        f.write(f"- End: {min_seg['end']}s\n")
        f.write(f"- Average Similarity: {min_sim:.4f}\n\n")

        # Add highest similarity patient segment (with >=30 windows) info
        f.write("## Highest Similarity Patient Segment (>=30 windows)\n\n")
        max_seg = stats['max_valid_seg']
        max_sim = stats['max_valid_sim']
        if max_seg is not None:
            f.write(f"The patient segment with at least 30 windows causing the highest similarity is:\n\n")
            f.write(f"- File: {max_seg['file']}\n")
            f.write(f"- Start: {max_seg['start']}s\n")
            f.write(f"- End: {max_seg['end']}s\n")
            f.write(f"- Average Similarity: {max_sim:.4f}\n\n")
        else:
            f.write("No patient segment with at least 30 windows found.\n\n")

        # Add highest similarity patient segments (total windows >= 30) info
        f.write("## Highest Similarity Patient Segments (Total Windows >= 30)\n\n")
        selected_patients = stats['selected_patients']
        selected_total_windows = stats['selected_total_windows']
        selected_max_seg = stats['selected_max_seg']
        selected_max_sim = stats['selected_max_sim']
        if selected_patients:
            f.write(f"Selected patient segments (by highest similarity) until total windows >= 30:\n\n")
            for seg in selected_patients:
                f.write(f"- File: {seg['file']}, Start: {seg['start']}s, End: {seg['end']}s\n")
            f.write(f"\nTotal windows: {selected_total_windows}\n")
            f.write(f"Highest similarity among these: {selected_max_sim:.4f} (File: {selected_max_seg['file']}, Start: {selected_max_seg['start']}s, End: {selected_max_seg['end']}s)\n\n")
        else:
            f.write("No patient segments found to reach at least 30 windows.\n\n")

        # Similarity without outliers (IQR method)
        f.write("## Mean Similarity Without Outliers (IQR Method)\n\n")
        filtered_stats = calculations.calculate_filtered_mean_similarity(stats['patients_avg'], stats['patients_segments'])
        f.write(f"Mean similarity (filtered): {filtered_stats['filtered_mean']:.4f}\n")
        f.write(f"Original mean: {filtered_stats['original_mean']:.4f}\n")
        f.write(f"Outlier threshold: <{filtered_stats['lower']:.4f}\n")
        f.write(f"Filtered count: {filtered_stats['filtered_count']}, Original count: {filtered_stats['original_count']}\n\n")
        if filtered_stats['outlier_segments']:
            f.write("### Discarded Patient Segments (Outliers)\n\n")
            for seg in filtered_stats['outlier_segments']:
                f.write(f"- File: {seg['file']}, Start: {seg['start']}s, End: {seg['end']}s, Avg Similarity: {seg.get('avg_similarity_to_icbhi', 'N/A')}\n")
            f.write("\n")
        # Overall Statistics (Filtered)
        filtered_values = [s for i, s in enumerate(stats['patients_avg']) if i not in filtered_stats['outlier_indices']]
        filtered_segments = [seg for i, seg in enumerate(stats['patients_segments']) if i not in filtered_stats['outlier_indices']]
        if filtered_values:
            f.write("## Overall Statistics (Filtered)\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Mean | {np.mean(filtered_values):.4f} |\n")
            f.write(f"| Median | {np.median(filtered_values):.4f} |\n")
            f.write(f"| Standard Deviation | {np.std(filtered_values):.4f} |\n")
            f.write(f"| Minimum | {np.min(filtered_values):.4f} |\n")
            f.write(f"| Maximum | {np.max(filtered_values):.4f} |\n\n")

            # Patients Segments Statistics (Filtered)
            f.write("## Patients Segments Statistics (Filtered)\n\n")
            f.write("Average similarity of each filtered patient segment to all ICBHI segments:\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Mean | {np.mean(filtered_values):.4f} |\n")
            f.write(f"| Min | {np.min(filtered_values):.4f} |\n")
            f.write(f"| Max | {np.max(filtered_values):.4f} |\n\n")

            # ICBHI Segments Statistics (Filtered)
            filtered_matrix = stats['matrix'][[i for i in range(len(stats['patients_avg'])) if i not in filtered_stats['outlier_indices']], :]
            filtered_icbhi_avg = np.mean(filtered_matrix, axis=0)
            f.write("## ICBHI Segments Statistics (Filtered)\n\n")
            f.write("Average similarity of each ICBHI segment to filtered patient segments:\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Mean | {np.mean(filtered_icbhi_avg):.4f} |\n")
            f.write(f"| Min | {np.min(filtered_icbhi_avg):.4f} |\n")
            f.write(f"| Max | {np.max(filtered_icbhi_avg):.4f} |\n\n")

            # Similarity Distribution (Filtered)
            f.write("## Similarity Distribution (Filtered)\n\n")
            f.write("| Range | Count | Percentage |\n")
            f.write("|-------|-------|------------|\n")
            ranges = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
            for low, high in ranges:
                count = np.sum((filtered_matrix >= low) & (filtered_matrix < high))
                percentage = count / filtered_matrix.size * 100
                f.write(f"| [{low:.2f}, {high:.2f}) | {count:,} | {percentage:.2f}% |\n")
            f.write("\n")

            # Least Similar Patient Segment (Filtered)
            min_idx = np.argmin(filtered_values)
            min_seg = filtered_segments[min_idx]
            min_sim = filtered_values[min_idx]
            f.write("## Least Similar Patient Segment (Filtered)\n\n")
            f.write(f"The filtered patient segment causing the least similarity is:\n\n")
            f.write(f"- File: {min_seg['file']}\n")
            f.write(f"- Start: {min_seg['start']}s\n")
            f.write(f"- End: {min_seg['end']}s\n")
            f.write(f"- Average Similarity: {min_sim:.4f}\n\n")

            # Highest Similarity Patient Segment (>=30 windows) (Filtered)
            patient_window_counts = [seg.get('window_count', 1) for seg in filtered_segments]
            valid_indices = [i for i, count in enumerate(patient_window_counts) if count >= 30]
            if valid_indices:
                max_valid_idx = valid_indices[np.argmax([filtered_values[i] for i in valid_indices])]
                max_valid_seg = filtered_segments[max_valid_idx]
                max_valid_sim = filtered_values[max_valid_idx]
                f.write("## Highest Similarity Patient Segment (>=30 windows, Filtered)\n\n")
                f.write(f"The filtered patient segment with at least 30 windows causing the highest similarity is:\n\n")
                f.write(f"- File: {max_valid_seg['file']}\n")
                f.write(f"- Start: {max_valid_seg['start']}s\n")
                f.write(f"- End: {max_valid_seg['end']}s\n")
                f.write(f"- Average Similarity: {max_valid_sim:.4f}\n\n")
            else:
                f.write("## Highest Similarity Patient Segment (>=30 windows, Filtered)\n\n")
                f.write("No filtered patient segment with at least 30 windows found.\n\n")

            # Highest Similarity Patient Segments (Total Windows >= 30) (Filtered)
            sorted_indices = np.argsort(filtered_values)[::-1]
            selected_indices = []
            total_windows = 0
            for idx in sorted_indices:
                count = filtered_segments[idx].get('window_count', 1)
                selected_indices.append(idx)
                total_windows += count
                if total_windows >= 30:
                    break
            if selected_indices:
                selected_patients = [filtered_segments[i] for i in selected_indices]
                selected_avg_sim = [filtered_values[i] for i in selected_indices]
                selected_total_windows = sum([seg.get('window_count', 1) for seg in selected_patients])
                selected_max_sim = max(selected_avg_sim)
                selected_max_seg = selected_patients[selected_avg_sim.index(selected_max_sim)]
                f.write("## Highest Similarity Patient Segments (Total Windows >= 30, Filtered)\n\n")
                f.write("Selected filtered patient segments (by highest similarity) until total windows >= 30:\n\n")
                for seg in selected_patients:
                    f.write(f"- File: {seg['file']}, Start: {seg['start']}s, End: {seg['end']}s\n")
                f.write(f"\nTotal windows: {selected_total_windows}\n")
                f.write(f"Highest similarity among these: {selected_max_sim:.4f} (File: {selected_max_seg['file']}, Start: {selected_max_seg['start']}s, End: {selected_max_seg['end']}s)\n\n")
            else:
                f.write("## Highest Similarity Patient Segments (Total Windows >= 30, Filtered)\n\n")
                f.write("No filtered patient segments found to reach at least 30 windows.\n\n")

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
                   'Patients Segments Count (Total)', 'ICBHI Segments Count', 'Total Pairs',
                   'Window Length (s)', 'Window Overlap (%)'],
        'Value': [stats['mean'], stats['median'], stats['std'],
                  stats['min'], stats['max'],
                  len(stats['patients_segments']), len(stats['icbhi_segments']),
                  stats['matrix'].size,
                  calculations.WINDOW_LENGTH, calculations.WINDOW_OVERLAP * 100]
    })
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary statistics saved to '{summary_csv}'")

    # Save distribution CSV
    dist_csv = os.path.join(output_dir, 'similarity_distribution.csv')
    dist_df = pd.DataFrame(stats['distribution'])
    dist_df.to_csv(dist_csv, index=False)
    print(f"Distribution saved to '{dist_csv}'")

    # Save patients segments CSV (no augmentation info)
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
    patients_segments, patients_original_count, patients_windowed_count = calculations.load_patients_healthy_sounds()

    # Load healthy sounds from ICBHI dataset
    print("\n[2/3] Loading healthy sounds from ICBHI dataset...")
    icbhi_segments, icbhi_original_count, icbhi_windowed_count = calculations.load_icbhi_healthy_sounds()

    # Calculate cosine similarity
    print("\n[3/3] Calculating cosine similarity...")
    result = calculations.calculate_cosine_similarity(patients_segments, icbhi_segments)

    if result is None:
        print("Cannot calculate similarity - missing data")
        return

    similarity_matrix, patients_segs, icbhi_segs = result

    # Analyze and report results
    stats = calculations.analyze_similarity(similarity_matrix, patients_segs, icbhi_segs)
    # Attach original and windowed counts for reporting
    stats['patients_original_count'] = patients_original_count
    stats['icbhi_original_count'] = icbhi_original_count
    stats['patients_windowed_count'] = patients_windowed_count
    stats['icbhi_windowed_count'] = icbhi_windowed_count

    # Generate comprehensive markdown report and CSV files
    generate_markdown_report(stats, 'reports')

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

