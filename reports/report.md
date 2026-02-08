# Cosine Similarity Analysis Report

## Healthy Sounds: Patients vs ICBHI Dataset

---

## Dataset Summary

- **Patients Segments:** 19
- **ICBHI Segments:** 3642
- **Total Pairs Analyzed:** 69,198

## Overall Statistics

| Metric | Value |
|--------|-------|
| Mean Similarity | 0.8937 |
| Median Similarity | 0.8992 |
| Standard Deviation | 0.0528 |
| Minimum Similarity | 0.4588 |
| Maximum Similarity | 0.9966 |

## Patients Segments Statistics

Average similarity of each patient segment to all ICBHI segments:

| Metric | Value |
|--------|-------|
| Mean | 0.8937 |
| Min | 0.8507 |
| Max | 0.9155 |

## ICBHI Segments Statistics

Average similarity of each ICBHI segment to all patient segments:

| Metric | Value |
|--------|-------|
| Mean | 0.8937 |
| Min | 0.6004 |
| Max | 0.9812 |

## Similarity Distribution

| Range | Count | Percentage |
|-------|-------|------------|
| [0.00, 0.25) | 0 | 0.00% |
| [0.25, 0.50) | 1 | 0.00% |
| [0.50, 0.75) | 1,103 | 1.59% |
| [0.75, 1.00) | 68,094 | 98.40% |

## Data Files

Detailed data is available in separate CSV files:

- `summary_statistics.csv` - Overall statistics
- `similarity_distribution.csv` - Distribution of similarity scores
- `patients_segments.csv` - Patient segment details with average similarities
- `icbhi_segments.csv` - ICBHI segment details with average similarities
- `all_pairs.csv` - All similarity pairs (sorted by similarity descending)

---

*Report generated automatically by cosine similarity analysis tool.*
