# ICBHI 2017 Challenge – Respiratory Sound Database

## Overview

This repository/document describes the **Respiratory Sound Database** released in conjunction with the **ICBHI 2017 Challenge** (International Conference on Biomedical and Health Informatics).

The database was compiled to support research on automatic respiratory sound analysis and is now **freely available for research purposes**. It includes both the **public and private datasets** used in the original challenge.

---

## Dataset Description

The Respiratory Sound Database contains audio recordings collected independently by two research teams across two countries over several years:

- **Portugal**  
  - School of Health Sciences, University of Aveiro (ESSUA)  
  - Respiratory Research and Rehabilitation Laboratory (Lab3R), ESSUA  
  - Hospital Infante D. Pedro, Aveiro  

- **Greece**  
  - Aristotle University of Thessaloniki (AUTH)  
  - University of Coimbra (UC)  
  - Papanikolaou General Hospital, Thessaloniki  
  - General Hospital of Imathia (Health Unit of Naousa)

### Dataset Statistics

- **Total duration:** ~5.5 hours of recordings  
- **Audio samples:** 920 annotated recordings  
- **Subjects:** 126  
- **Respiratory cycles:** 6,898 total  
  - 1,864 with crackles  
  - 886 with wheezes  
  - 506 with both crackles and wheezes  

Recordings vary in duration from **10s to 90s**, were acquired using **heterogeneous equipment**, and include **realistic noise levels** to reflect real-world clinical conditions.

---

## Annotations

Each respiratory cycle was annotated by respiratory experts with the following labels:

- Crackles
- Wheezes
- Both crackles and wheezes
- No adventitious sounds

### Annotation File Format

Each annotation file contains **four columns**:

1. Beginning of respiratory cycle (seconds)
2. End of respiratory cycle (seconds)
3. Crackles presence (1 = present, 0 = absent)
4. Wheezes presence (1 = present, 0 = absent)

---

## File Naming Convention

Each audio filename consists of **five underscore-separated fields**:

```

PatientID_RecordingIndex_ChestLocation_AcquisitionMode_Equipment.wav

```

### 1. Patient Number
- Ranges from `101` to `226`

### 2. Recording Index
- Integer identifying the recording session

### 3. Chest Location
- `Tc` – Trachea  
- `Al` – Anterior Left  
- `Ar` – Anterior Right  
- `Pl` – Posterior Left  
- `Pr` – Posterior Right  
- `Ll` – Lateral Left  
- `Lr` – Lateral Right  

### 4. Acquisition Mode
- `sc` – Sequential / Single-channel  
- `mc` – Simultaneous / Multi-channel  

### 5. Recording Equipment
- `AKGC417L` – AKG C417L Microphone  
- `LittC2SE` – 3M Littmann Classic II SE  
- `Litt3200` – 3M Littmann 3200 Electronic Stethoscope  
- `Meditron` – Welch Allyn Meditron Master Elite  

---

## Diagnostic Labels

Diagnostic information for each subject is provided separately.

### Diagnosis Abbreviations

- **COPD** – Chronic Obstructive Pulmonary Disease  
- **LRTI** – Lower Respiratory Tract Infection  
- **URTI** – Upper Respiratory Tract Infection  

- Diagnosis file:  
  https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_Challenge_diagnosis.txt

- Train/test split file:  
  https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_challenge_train_test.txt

---

## Additional Metadata

- **Demographic information:**  
  Participant ID, Age, Sex, Adult BMI (kg/m²), Child Weight (kg), Child Height (cm)  
  https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_Challenge_demographic_information.txt

- **Detailed respiratory events:**  
  https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/events.zip

---

## Download

- **Full database (ZIP):**  
  https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip

- **Browse files online:**  
  https://bhichallenge.med.auth.gr/node/51

---

## Important Notice (Filename Correction)

Due to a bug in an earlier release, **92 files contained incorrect recording equipment identifiers** in their filenames.

- The **current version** of the database has corrected filenames.
- Differences between old and corrected filenames:  
  https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/filename_differences.txt
- A bash script for automatic renaming is also provided by the authors.

---

## Citation

Publications using this database **must cite** the following paper:

> Rocha BM et al. (2019).  
> *An open access database for the evaluation of respiratory sound classification algorithms.*  
> Physiological Measurement, 40(3), 035001.

---

## Contact

Feedback and comments on the dataset are welcomed:

- **Email:** icbhi_challenge@med.auth.gr

---

## License and Usage

This database is **freely available for research purposes**.  
Users are responsible for ensuring compliance with applicable ethical and institutional guidelines when using the data.

---

## Cosine Similarity Analysis: Patients vs ICBHI Healthy Sounds

This section explains how we compared healthy respiratory sounds from our local **patients dataset** against the **ICBHI dataset** to measure their similarity.

### Goal

We wanted to answer: *"How similar are the healthy breathing sounds recorded from our patients to the healthy breathing sounds in the established ICBHI database?"*

This helps validate whether our patient recordings are comparable to the reference dataset used in respiratory sound research.

---

### Step 1: Identifying Healthy Sounds

#### From the Patients Dataset
- We looked at diagnosis CSV files created by doctors (Dr. Fatih and Dr. Guney)
- Each CSV file contains time segments with diagnoses like "normal", "ral" (rales), "ronkus" (rhonchi), etc.
- We extracted only the segments labeled as **"normal"** (healthy breathing sounds)
- Result: **19 unique healthy segments** from various patient recordings

#### From the ICBHI Dataset
- Each audio file has a corresponding `.txt` annotation file
- The annotation file contains respiratory cycles with columns: `start_time`, `end_time`, `crackles (0/1)`, `wheezes (0/1)`
- We extracted segments where **crackles = 0 AND wheezes = 0** (no abnormal sounds)
- Result: **3,642 healthy segments** from 920 recordings

---

### Step 2: The Length Problem

We discovered a significant challenge:

| Dataset | Segment Count | Duration Range | Mean | Std Dev |
|---------|---------------|----------------|------|---------|
| Patients | 19 | 0.42s – 16.67s | 3.36s | **4.45s** |
| ICBHI | 3,642 | 0.20s – 16.16s | 2.61s | **1.28s** |

**Problem:** Patient segments had highly variable lengths (some less than 1 second, others over 16 seconds), making direct comparison unfair. A 16-second segment contains very different information than a 0.5-second segment.

---

### Step 3: Normalizing Segment Lengths (Windowing)

To make fair comparisons, we split all segments into **fixed-length windows**:

- **Window length:** 2.0 seconds (chosen to match ICBHI's median duration of ~2.44s)
- **Overlap:** 50% (each window shares half its content with the next)
- **Minimum segment:** 0.5 seconds (shorter segments are discarded)

**How it works:**
```
Original segment: |-------- 6 seconds --------|

Split into windows with 50% overlap:
Window 1: |-- 2s --|
Window 2:     |-- 2s --|
Window 3:         |-- 2s --|
Window 4:             |-- 2s --|
Window 5:                 |-- 2s --|

Result: 5 windows from one 6-second segment
```

**Short segments (< 2 seconds):**
- Padded with zeros to reach 2 seconds
- Example: A 1.5-second segment becomes a 2-second segment with 0.5 seconds of silence

**Results after windowing:**
- Patients: 19 segments → **50 windows**
- ICBHI: 3,642 segments → **5,489 windows**

---

### Step 4: Data Augmentation (Increasing Patient Samples)

With only 50 patient windows vs 5,489 ICBHI windows, we needed more patient samples for robust comparison. We applied **audio augmentation** to create variations of each patient window.

**Augmentation techniques (5 variations per window):**

1. **Pitch Shift Up** – Raises the pitch by 1-3 semitones (like a higher-pitched voice)
2. **Pitch Shift Down** – Lowers the pitch by 1-3 semitones (like a deeper voice)
3. **Time Stretch (Faster)** – Speeds up by 5-10% without changing pitch
4. **Time Stretch (Slower)** – Slows down by 5-10% without changing pitch
5. **Noise Addition** – Adds subtle white noise to simulate real-world recording conditions

Each augmented version also gets a random volume adjustment (80-120% of original).

**Result:** 50 original windows × 6 (1 original + 5 augmented) = **300 patient segments**

---

### Step 5: Feature Extraction

Raw audio waveforms can't be directly compared. We need to convert each 2-second audio window into a **numerical fingerprint** (feature vector) that captures its acoustic characteristics.

**Features extracted for each window:**

| Feature Type | Description | Values |
|--------------|-------------|--------|
| **MFCCs** (Mean) | Mel-frequency cepstral coefficients – captures the "shape" of the sound spectrum, similar to how human ears perceive sound | 13 values |
| **MFCCs** (Std Dev) | Variation in MFCCs over time – captures how the sound changes | 13 values |
| **Spectral Centroid** | The "center of mass" of the sound spectrum – indicates brightness | 1 value |
| **Spectral Bandwidth** | How spread out the frequencies are | 1 value |
| **Spectral Rolloff** | Frequency below which 85% of the energy exists | 1 value |
| **Zero Crossing Rate** | How often the signal crosses zero – indicates noisiness | 1 value |

**Total: 30 features per audio window**

Each audio window is now represented as a vector of 30 numbers, like:
```
[2.34, -1.56, 0.89, ..., 1456.2, 3421.5, 2890.1, 0.087]
```

---

### Step 6: Calculating Cosine Similarity

**What is Cosine Similarity?**

Cosine similarity measures how similar two vectors are by looking at the angle between them, ignoring their magnitude. It ranges from -1 to 1:
- **1.0** = Identical direction (perfectly similar)
- **0.0** = Perpendicular (no similarity)
- **-1.0** = Opposite direction (completely opposite)

**Formula:**
```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)

Where:
- A · B = dot product of vectors A and B
- ||A|| = magnitude (length) of vector A
```

**Example:**
```
Patient window features: [0.5, 0.3, 0.8, ...]
ICBHI window features:   [0.6, 0.4, 0.7, ...]

If these point in nearly the same direction → high similarity (~0.95)
If these point in different directions → low similarity (~0.30)
```

**What we calculated:**
- Every patient window (300) compared to every ICBHI window (5,489)
- Total comparisons: **300 × 5,489 = 1,646,700 pairs**

---

### Step 7: Results

#### Overall Similarity Statistics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Mean** | 0.8707 | On average, patient sounds are 87% similar to ICBHI sounds |
| **Median** | 0.8963 | Half of all pairs have similarity > 90% |
| **Std Dev** | 0.0998 | Most similarities cluster around the mean |
| **Minimum** | 0.2134 | Some pairs are quite different |
| **Maximum** | 0.9999 | Some pairs are nearly identical |

#### Similarity Distribution

| Similarity Range | Pairs | Percentage | Meaning |
|------------------|-------|------------|---------|
| 0.00 – 0.25 | 27 | 0.00% | Very different |
| 0.25 – 0.50 | 17,071 | 1.04% | Somewhat different |
| 0.50 – 0.75 | 154,599 | 9.39% | Moderately similar |
| **0.75 – 1.00** | **1,475,003** | **89.57%** | **Highly similar** |

**Key Finding:** Nearly 90% of all patient-ICBHI pairs have similarity ≥ 0.75, indicating that our patient recordings are acoustically very similar to the ICBHI reference dataset.

#### Top 5 Most Similar Pairs (Original Segments Only)

| Rank | Similarity | Patient File | Patient Time | ICBHI File | ICBHI Time |
|------|------------|--------------|--------------|------------|------------|
| 1 | 0.9976 | 006b5409921f4b0ebf501a7d6edaef45 | 14.11s – 16.11s | 198_6p1_Al_mc_AKGC417L | 18.82s – 20.82s |
| 2 | 0.9973 | a04dc9d0c8e34418a742731030fe259e | 8.69s – 10.69s | 113_1b1_Ar_sc_Litt3200 | 1.15s – 3.15s |
| 3 | 0.9967 | a04dc9d0c8e34418a742731030fe259e | 8.69s – 10.69s | 124_1b1_Pl_sc_Litt3200 | 5.81s – 7.81s |
| 4 | 0.9966 | 006b5409921f4b0ebf501a7d6edaef45 | 14.11s – 16.11s | 159_1b1_Ll_sc_Meditron | 18.84s – 20.84s |
| 5 | 0.9965 | a04dc9d0c8e34418a742731030fe259e | 8.69s – 10.69s | 185_1b1_Ar_sc_Litt3200 | 2.34s – 4.34s |

---

### Generated Reports

The analysis generates the following files in the `reports/` folder:

| File | Description |
|------|-------------|
| `report.md` | Summary report in Markdown format |
| `summary_statistics.csv` | Overall statistics and configuration |
| `similarity_distribution.csv` | Distribution of similarity scores |
| `patients_segments.csv` | All patient segments with their average similarity to ICBHI |
| `icbhi_segments.csv` | All ICBHI segments with their average similarity to patients |
| `all_pairs.csv` | All 1.6M similarity pairs (sorted by similarity) |
| `all_pairs_original_only.csv` | Pairs using only original (non-augmented) patient segments |

---

### Running the Analysis

```bash
# Install dependencies
pip install -r requirements.txt

# Run the analysis
python main.py
```

**Requirements:**
- Python 3.8+
- librosa (audio processing)
- numpy, pandas (data handling)
- scikit-learn (cosine similarity)
- tqdm (progress bars)

---

### Configuration

You can modify these parameters in `main.py`:

```python
WINDOW_LENGTH = 2.0          # Window size in seconds
WINDOW_OVERLAP = 0.5         # 50% overlap between windows
MIN_SEGMENT_LENGTH = 0.5     # Minimum segment length to process
ENABLE_AUGMENTATION = True   # Enable/disable data augmentation
AUGMENTATION_MULTIPLIER = 5  # Number of augmented versions per segment
SAMPLE_RATE = 22050          # Audio sample rate in Hz
```

---

### Conclusion

The cosine similarity analysis shows that **healthy respiratory sounds from our patients dataset are highly similar to healthy sounds in the ICBHI reference database**, with ~90% of comparisons yielding similarity scores above 0.75. This validates that our patient recordings are consistent with established respiratory sound datasets and suitable for further analysis or machine learning applications.

