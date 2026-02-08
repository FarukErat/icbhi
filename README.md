# ICBHI 2017 Challenge
## Respiratory Sound Database

## Overview
The **Respiratory Sound Database** was originally compiled to support the scientific challenge organized at the *International Conference on Biomedical Health Informatics (ICBHI) 2017*.  
The current version is freely available for research and includes both the **public** and **private** datasets of the ICBHI Challenge.

---

## Data Collection
Audio samples were collected independently by two research teams in two different countries over several years.

### Portugal
- School of Health Sciences, University of Aveiro (ESSUA)
- Respiratory Research and Rehabilitation Laboratory (Lab3R), ESSUA
- Hospital Infante D. Pedro, Aveiro

### Greece
- Aristotle University of Thessaloniki (AUTH)
- University of Coimbra (UC)
- Papanikolaou General Hospital, Thessaloniki
- General Hospital of Imathia (Health Unit of Naousa)

Recordings were acquired using heterogeneous equipment and reflect real clinical environments, including high-noise conditions.

---

## Dataset Statistics
- **Total duration:** ~5.5 hours  
- **Respiratory cycles:** 6,898  
  - Crackles: 1,864  
  - Wheezes: 886  
  - Crackles + Wheezes: 506  
- **Annotated recordings:** 920  
- **Subjects:** 126  
- **Recording length:** 10–90 seconds  

---

## Annotations
Respiratory cycles were annotated by respiratory experts as:
- Crackles
- Wheezes
- Crackles and wheezes
- No adventitious sounds

### Annotation File Format
Each annotation file contains four columns:
1. Beginning of respiratory cycle(s)
2. End of respiratory cycle(s)
3. Crackles (`1` = present, `0` = absent)
4. Wheezes (`1` = present, `0` = absent)

---

## File Naming Convention
Each audio file name consists of **five underscore-separated fields**:

1. **Patient number**  
   `101, 102, ..., 226`

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

## Diagnosis Labels
Diagnosis information is provided per subject.

Abbreviations:
- **COPD** – Chronic Obstructive Pulmonary Disease  
- **LRTI** – Lower Respiratory Tract Infection  
- **URTI** – Upper Respiratory Tract Infection  

- Diagnosis file:  
  https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_Challenge_diagnosis.txt
- Train/test split:  
  https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_challenge_train_test.txt

---

## Demographic Information
Demographic metadata is provided with the following columns:
1. Participant ID  
2. Age  
3. Sex  
4. Adult BMI (kg/m²)  
5. Child Weight (kg)  
6. Child Height (cm)  

`NA` indicates *Not Available*.

- File:  
  https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_Challenge_demographic_information.txt

---

## Download
The dataset is **freely available for research**.

- Full dataset (ZIP):  
  https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip
- Browse files online:  
  https://bhichallenge.med.auth.gr/node/51
- Detailed respiratory events:  
  https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/events.zip

---

## Known File Naming Issue
Due to a bug in an early release:
- **92 files** had incorrect recording equipment identifiers.

The corrected filenames are included in the updated dataset.

- Filename differences:  
  https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/filename_differences.txt

If using an older version, replace the last 8 characters of affected filenames with `Meditron`, or use the provided bash script.

---

## Citation
If you use this database, **please cite**:

> Rocha BM et al. (2019)  
> *An open access database for the evaluation of respiratory sound classification algorithms*  
> Physiological Measurement, 40, 035001

---

## Contact
Feedback and comments are welcome:

**Email:** icbhi_challenge@med.auth.gr
