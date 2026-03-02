# Cosine Similarity Analysis Report

## Healthy Sounds: Patients vs ICBHI Dataset

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Window Length | 2.0s |
| Window Overlap | 0% |
| Min Segment Length | 0.5s |
| Sample Rate | 22050 Hz |

## Dataset Summary

| Dataset | Original Segments | Windowed Segments | Total |
|---------|-------------------|-------------------|-------|
| Patients | 19 | 33 | 33 |
| ICBHI | 3642 | 4114 | 4114 |

**Total Pairs Analyzed:** 135,762

## Overall Statistics

| Metric | Value |
|--------|-------|
| Mean Similarity | 0.8924 |
| Median Similarity | 0.9089 |
| Standard Deviation | 0.0729 |
| Minimum Similarity | 0.4962 |
| Maximum Similarity | 0.9976 |

## Patients Segments Statistics

Average similarity of each patient segment to all ICBHI segments:

| Metric | Value |
|--------|-------|
| Mean | 0.8924 |
| Min | 0.7741 |
| Max | 0.9232 |

## ICBHI Segments Statistics

Average similarity of each ICBHI segment to all patient segments:

| Metric | Value |
|--------|-------|
| Mean | 0.8924 |
| Min | 0.7181 |
| Max | 0.9758 |

## Similarity Distribution

| Range | Count | Percentage |
|-------|-------|------------|
| [0.00, 0.25) | 0 | 0.00% |
| [0.25, 0.50) | 6 | 0.00% |
| [0.50, 0.75) | 7,656 | 5.64% |
| [0.75, 1.00) | 128,100 | 94.36% |

## Least Similar Patient Segment

The patient segment causing the least similarity is:

- File: 4aff9316a1dd4c96b26a187b6e5b9ed6
- Start: 3.827722s
- End: 5.827722s
- Average Similarity: 0.7741

## Highest Similarity Patient Segment (>=30 windows)

No patient segment with at least 30 windows found.

## Highest Similarity Patient Segments (Total Windows >= 30)

Selected patient segments (by highest similarity) until total windows >= 30:

- File: fd2936a2bed74e97b92d66b0cff2db2a, Start: 15.210671s, End: 17.210671s
- File: fa76fcbd05634e5ba97ef61a6803df61, Start: 7.79447s, End: 9.79447s
- File: 006b5409921f4b0ebf501a7d6edaef45, Start: 9.50753s, End: 11.50753s
- File: fd2936a2bed74e97b92d66b0cff2db2a, Start: 5.210671s, End: 7.210671s
- File: fa76fcbd05634e5ba97ef61a6803df61, Start: 5.285432s, End: 7.285432s
- File: fd2936a2bed74e97b92d66b0cff2db2a, Start: 13.210671s, End: 15.210671s
- File: 2b02880eda824aafbe536fb680c0f9df, Start: 5.342606s, End: 7.342606s
- File: fd2936a2bed74e97b92d66b0cff2db2a, Start: 3.210671s, End: 5.210671s
- File: 006b5409921f4b0ebf501a7d6edaef45, Start: 26.997026s, End: 28.997026s
- File: fd2936a2bed74e97b92d66b0cff2db2a, Start: 9.210671s, End: 11.210671s
- File: fd2936a2bed74e97b92d66b0cff2db2a, Start: 7.210671s, End: 9.210671s
- File: 2b02880eda824aafbe536fb680c0f9df, Start: 3.342606s, End: 5.342606s
- File: 9e756a203d054ca1918d5c9225cff153, Start: 10.252022s, End: 12.252022s
- File: 3ae2c9de644647368ab4316cb60374f0, Start: 10.405755s, End: 12.405755s
- File: 9e756a203d054ca1918d5c9225cff153, Start: 8.252022s, End: 10.252022s
- File: a04dc9d0c8e34418a742731030fe259e, Start: 14.285141s, End: 16.285141s
- File: 9e756a203d054ca1918d5c9225cff153, Start: 2.252022s, End: 4.252022s
- File: 9e756a203d054ca1918d5c9225cff153, Start: 6.252022s, End: 8.252022s
- File: fd2936a2bed74e97b92d66b0cff2db2a, Start: 1.210671s, End: 3.210671s
- File: fd2936a2bed74e97b92d66b0cff2db2a, Start: 11.210671s, End: 13.210671s
- File: 9e756a203d054ca1918d5c9225cff153, Start: 4.252022s, End: 6.252022s
- File: 9e756a203d054ca1918d5c9225cff153, Start: 12.252022s, End: 14.252022s
- File: 9e756a203d054ca1918d5c9225cff153, Start: 14.252022s, End: 16.252022s
- File: 006b5409921f4b0ebf501a7d6edaef45, Start: 14.111463s, End: 16.111463s
- File: 2c5f90f58c0643bfa7d960ca1be51dfa, Start: 0.777554s, End: 2.777554s
- File: 2b02880eda824aafbe536fb680c0f9df, Start: 1.342606s, End: 3.342606s
- File: f413e415b2de48ffaf0f0ada8eaa248c, Start: 26.511172s, End: 28.511172s
- File: 899b8c8fc465490ca9710ffd7508bb83, Start: 6.261743s, End: 8.261743s
- File: fa76fcbd05634e5ba97ef61a6803df61, Start: 3.434503s, End: 5.434502999999999s
- File: f413e415b2de48ffaf0f0ada8eaa248c, Start: 6.413517s, End: 8.413516999999999s

Total windows: 30
Highest similarity among these: 0.9232 (File: fd2936a2bed74e97b92d66b0cff2db2a, Start: 15.210671s, End: 17.210671s)

## Mean Similarity Without Outliers (IQR Method)

Mean similarity (filtered): 0.9082
Original mean: 0.8924
Outlier threshold: <0.8723
Filtered count: 27, Original count: 33

### Discarded Patient Segments (Outliers)

- File: f413e415b2de48ffaf0f0ada8eaa248c, Start: 6.413517s, End: 8.413516999999999s, Avg Similarity: N/A
- File: fa76fcbd05634e5ba97ef61a6803df61, Start: 3.434503s, End: 5.434502999999999s, Avg Similarity: N/A
- File: 4aff9316a1dd4c96b26a187b6e5b9ed6, Start: 1.446523s, End: 3.446523s, Avg Similarity: N/A
- File: 4aff9316a1dd4c96b26a187b6e5b9ed6, Start: 3.827722s, End: 5.827722s, Avg Similarity: N/A
- File: a04dc9d0c8e34418a742731030fe259e, Start: 8.68553s, End: 10.68553s, Avg Similarity: N/A
- File: 899b8c8fc465490ca9710ffd7508bb83, Start: 6.261743s, End: 8.261743s, Avg Similarity: N/A

## Overall Statistics (Filtered)

| Metric | Value |
|--------|-------|
| Mean | 0.9082 |
| Median | 0.9093 |
| Standard Deviation | 0.0099 |
| Minimum | 0.8793 |
| Maximum | 0.9232 |

## Patients Segments Statistics (Filtered)

Average similarity of each filtered patient segment to all ICBHI segments:

| Metric | Value |
|--------|-------|
| Mean | 0.9082 |
| Min | 0.8793 |
| Max | 0.9232 |

## ICBHI Segments Statistics (Filtered)

Average similarity of each ICBHI segment to filtered patient segments:

| Metric | Value |
|--------|-------|
| Mean | 0.9082 |
| Min | 0.6991 |
| Max | 0.9874 |

## Similarity Distribution (Filtered)

| Range | Count | Percentage |
|-------|-------|------------|
| [0.00, 0.25) | 0 | 0.00% |
| [0.25, 0.50) | 0 | 0.00% |
| [0.50, 0.75) | 968 | 0.87% |
| [0.75, 1.00) | 110,110 | 99.13% |

## Least Similar Patient Segment (Filtered)

The filtered patient segment causing the least similarity is:

- File: f413e415b2de48ffaf0f0ada8eaa248c
- Start: 26.511172s
- End: 28.511172s
- Average Similarity: 0.8793

## Highest Similarity Patient Segment (>=30 windows, Filtered)

No filtered patient segment with at least 30 windows found.

## Highest Similarity Patient Segments (Total Windows >= 30, Filtered)

Selected filtered patient segments (by highest similarity) until total windows >= 30:

- File: fd2936a2bed74e97b92d66b0cff2db2a, Start: 15.210671s, End: 17.210671s
- File: fa76fcbd05634e5ba97ef61a6803df61, Start: 7.79447s, End: 9.79447s
- File: 006b5409921f4b0ebf501a7d6edaef45, Start: 9.50753s, End: 11.50753s
- File: fd2936a2bed74e97b92d66b0cff2db2a, Start: 5.210671s, End: 7.210671s
- File: fa76fcbd05634e5ba97ef61a6803df61, Start: 5.285432s, End: 7.285432s
- File: fd2936a2bed74e97b92d66b0cff2db2a, Start: 13.210671s, End: 15.210671s
- File: 2b02880eda824aafbe536fb680c0f9df, Start: 5.342606s, End: 7.342606s
- File: fd2936a2bed74e97b92d66b0cff2db2a, Start: 3.210671s, End: 5.210671s
- File: 006b5409921f4b0ebf501a7d6edaef45, Start: 26.997026s, End: 28.997026s
- File: fd2936a2bed74e97b92d66b0cff2db2a, Start: 9.210671s, End: 11.210671s
- File: fd2936a2bed74e97b92d66b0cff2db2a, Start: 7.210671s, End: 9.210671s
- File: 2b02880eda824aafbe536fb680c0f9df, Start: 3.342606s, End: 5.342606s
- File: 9e756a203d054ca1918d5c9225cff153, Start: 10.252022s, End: 12.252022s
- File: 3ae2c9de644647368ab4316cb60374f0, Start: 10.405755s, End: 12.405755s
- File: 9e756a203d054ca1918d5c9225cff153, Start: 8.252022s, End: 10.252022s
- File: a04dc9d0c8e34418a742731030fe259e, Start: 14.285141s, End: 16.285141s
- File: 9e756a203d054ca1918d5c9225cff153, Start: 2.252022s, End: 4.252022s
- File: 9e756a203d054ca1918d5c9225cff153, Start: 6.252022s, End: 8.252022s
- File: fd2936a2bed74e97b92d66b0cff2db2a, Start: 1.210671s, End: 3.210671s
- File: fd2936a2bed74e97b92d66b0cff2db2a, Start: 11.210671s, End: 13.210671s
- File: 9e756a203d054ca1918d5c9225cff153, Start: 4.252022s, End: 6.252022s
- File: 9e756a203d054ca1918d5c9225cff153, Start: 12.252022s, End: 14.252022s
- File: 9e756a203d054ca1918d5c9225cff153, Start: 14.252022s, End: 16.252022s
- File: 006b5409921f4b0ebf501a7d6edaef45, Start: 14.111463s, End: 16.111463s
- File: 2c5f90f58c0643bfa7d960ca1be51dfa, Start: 0.777554s, End: 2.777554s
- File: 2b02880eda824aafbe536fb680c0f9df, Start: 1.342606s, End: 3.342606s
- File: f413e415b2de48ffaf0f0ada8eaa248c, Start: 26.511172s, End: 28.511172s

Total windows: 27
Highest similarity among these: 0.9232 (File: fd2936a2bed74e97b92d66b0cff2db2a, Start: 15.210671s, End: 17.210671s)

## Data Files

Detailed data is available in separate CSV files:

- `summary_statistics.csv` - Overall statistics
- `similarity_distribution.csv` - Distribution of similarity scores
- `patients_segments.csv` - Patient segment details with average similarities
- `icbhi_segments.csv` - ICBHI segment details with average similarities
- `all_pairs.csv` - All similarity pairs (sorted by similarity descending)
- `all_pairs_original_only.csv` - Pairs with original (non-augmented) patient segments only

---

*Report generated automatically by cosine similarity analysis tool.*
