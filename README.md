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
