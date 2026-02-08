# ICBHI 2017 Challenge – Respiratory Sound Database

## Overview
The **Respiratory Sound Database** was originally compiled to support the scientific challenge organized at the *International Conference on Biomedical Health Informatics (ICBHI) 2017*.  
The current version is freely available for research purposes and includes both the **public** and **private** datasets of the ICBHI Challenge.

This database is intended for the development and evaluation of algorithms for respiratory sound analysis, particularly for the detection of adventitious sounds such as **crackles** and **wheezes**.

---

## Data Collection
Audio samples were collected independently by two research teams in two different countries over several years:

- **Portugal**  
  School of Health Sciences, University of Aveiro (ESSUA)  
  Respiratory Research and Rehabilitation Laboratory (Lab3R), ESSUA  
  Hospital Infante D. Pedro, Aveiro

- **Greece**  
  Aristotle University of Thessaloniki (AUTH)  
  University of Coimbra (UC)  
  Papanikolaou General Hospital, Thessaloniki  
  General Hospital of Imathia (Health Unit of Naousa)

Recordings were acquired using heterogeneous equipment and under real clinical conditions, including cases with high noise levels.

---

## Dataset Statistics
- **Total duration:** ~5.5 hours  
- **Total respiratory cycles:** 6,898  
  - Crackles: 1,864  
  - Wheezes: 886  
  - Crackles + Wheezes: 506  
- **Audio samples:** 920 annotated recordings  
- **Subjects:** 126  
- **Recording duration:** 10–90 seconds per file  

---

## Annotations
Respiratory cycles were annotated by respiratory experts. Each cycle is labeled as containing:
- Crackles
- Wheezes
- Both crackles and wheezes
- No adventitious respiratory sounds

### Annotation File Format
Each annotation file contains four columns:
1. Beginning of respiratory cycle(s)
2. End of respiratory cycle(s)
3. Presence of crackles (`1` = present, `0` = absent)
4. Presence of wheezes (`1` = present, `0` = absent)

---

## File Naming Convention
Each audio file name is composed of **five elements**, separated by underscores (`_`):

1. **Patient number**  
   Example: `101, 102, ..., 226`

2. **Recording index**

3. **Chest location**
   - `Tc` – Trachea  
   - `Al` – Anterior left  
   - `Ar` – Anterior right  
   - `Pl` – Posterior left  
   - `Pr` – Posterior right  
   - `Ll` – Lateral left  
   - `Lr` – Lateral right  

4. **Acquisition mode**
   - `sc` – Sequential / single channel  
   - `mc` – Simultaneous / multichannel  

5. **Recording equipment**
   - `AKGC417L` – AKG C417L Microphone  
   - `LittC2SE` – 3M Littmann Classic II SE Stethoscope  
   - `Litt3200` – 3M Littmann 3200 Electronic Stethoscope  
   - `Meditron` – Welch Allyn Meditron Master Elite Electronic Stethoscope  

---

## Diagnosis Information
Diagnosis labels are provided per subject. The abbreviations used are:

- **COPD** – Chronic Obstructive Pulmonary Disease  
- **LRTI** – Lower Respiratory Tract Infection  
- **URTI** – Upper Respiratory Tract Infection  

Files indicating:
- Subject diagnoses
- Training/test split distribution  

are provided alongside the dataset.

---

## Demographic Information
Additional files contain demographic and clinical metadata. Columns correspond to:

1. Participant ID  
2. Age  
3. Sex  
4. Adult BMI (kg/m²)  
5. Child Weight (kg)  
6. Child Height (cm)  

`NA` indicates *Not Available*.

---

## Known File Naming Issue
Due to a bug in an early release:
- **92 files** had incorrect recording equipment identifiers in their file names.

The corrected filenames are included in the updated dataset.  
Differences can be found in `filename_differences.txt`.

To fix older versions:
- Replace the last 8 characters of the filename with `Meditron`, **or**
- Use the provided bash script to rename files automatically.

---

## Usage and License
This database is **freely available for research purposes**.  
Users are encouraged to share feedback and experiences using the dataset.

---

## Citation
Publications using this database **must cite** the following paper:

> Rocha BM et al. (2019).  
> *An open access database for the evaluation of respiratory sound classification algorithms*.  
> **Physiological Measurement**, 40, 035001.

---

## Contact
For comments, feedback, or questions regarding the dataset:

**Email:** `icbhi_challenge@med.auth.gr`
