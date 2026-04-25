# ICBHI 2017 Respiratory Sound Challenge

> **International Conference on Biomedical and Health Informatics — Scientific Challenge**
> Aristotle University of Thessaloniki (AUTH) · University of Coimbra (UC) · University of Aveiro (ESSUA)

---

## Table of Contents

- [Overview](#overview)
- [Dataset Summary](#dataset-summary)
- [Data Collection](#data-collection)
- [Database Structure](#database-structure)
  - [File Naming Convention](#file-naming-convention)
  - [Annotation Format](#annotation-format)
  - [Demographic & Diagnosis Files](#demographic--diagnosis-files)
- [Sound Categories](#sound-categories)
- [Recording Equipment](#recording-equipment)
- [Chest Locations](#chest-locations)
- [Train / Test Split](#train--test-split)
- [Evaluation Metrics](#evaluation-metrics)
- [Citation](#citation)
- [Contact](#contact)
- [License](#license)

---

## Overview

The **ICBHI 2017 Respiratory Sound Database** was originally compiled to support the scientific challenge organized at the *International Conference on Biomedical and Health Informatics (ICBHI) 2017*. The current version is **freely available** for research and contains both the public and private datasets of the original challenge.

The database provides a benchmark resource for the development and evaluation of **automated respiratory sound classification** algorithms, targeting the detection of pathological breath sounds such as crackles and wheezes.

🔗 **Official page:** https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
📧 **Contact:** icbhi_challenge@med.auth.gr

---

## Dataset Summary

| Attribute | Value |
|---|---|
| Total recording duration | **5.5 hours** |
| Total audio samples | **920** |
| Total annotated respiratory cycles | **6,898** |
| Number of subjects | **126** (79 adults, 46 children, 1 unknown) |
| Normal cycles | 3,643 |
| Cycles with crackles | 1,864 |
| Cycles with wheezes | 886 |
| Cycles with both crackles & wheezes | 506 |
| Recording duration per file | 10 s – 90 s |

---

## Data Collection

Audio samples were collected **independently** by two research teams across two countries over several years:

- **Team 1 — Portugal**
  School of Health Sciences, University of Aveiro (ESSUA)
  - *Respiratory Research and Rehabilitation Laboratory (Lab3R), ESSUA*
  - *Hospital Infante D. Pedro, Aveiro*

- **Team 2 — Greece**
  Aristotle University of Thessaloniki (AUTH) & University of Coimbra (UC)
  - *Papanikolaou General Hospital, Thessaloniki*
  - *General Hospital of Imathia (Health Unit of Naousa)*

Recordings were collected using **heterogeneous equipment** in both clinical and non-clinical environments, introducing realistic noise levels (including heartbeat and environmental noise) that simulate real-life auscultation conditions.

---

## Database Structure

### File Naming Convention

Each audio filename is composed of **5 elements** separated by underscores (`_`):

```
<patient_id>_<recording_index>_<chest_location>_<acquisition_mode>_<recording_equipment>
```

| Element | Description |
|---|---|
| `patient_id` | Unique subject identifier |
| `recording_index` | Index of the recording for that patient |
| `chest_location` | Anatomical location on the chest (see [Chest Locations](#chest-locations)) |
| `acquisition_mode` | `sc` = single-channel · `mc` = multi-channel |
| `recording_equipment` | Device used (see [Recording Equipment](#recording-equipment)) |

**Example:** `101_1b1_Al_sc_Meditron`

### Annotation Format

Each audio file (`.wav`) is accompanied by a plain-text annotation file (`.txt`) with one row per respiratory cycle:

```
<start_time>  <end_time>  <crackles>  <wheezes>
```

| Column | Type | Description |
|---|---|---|
| `start_time` | float (s) | Beginning of the respiratory cycle |
| `end_time` | float (s) | End of the respiratory cycle |
| `crackles` | 0 / 1 | Presence of crackles |
| `wheezes` | 0 / 1 | Presence of wheezes |

This yields four possible cycle classes:

| Crackles | Wheezes | Label |
|:---:|:---:|---|
| 0 | 0 | Normal |
| 1 | 0 | Crackles |
| 0 | 1 | Wheezes |
| 1 | 1 | Crackles + Wheezes |

### Demographic & Diagnosis Files

| File | Contents |
|---|---|
| `demographic_info.txt` | Age, sex, BMI, weight, height per subject |
| `diagnosis.txt` | Pathological diagnosis label per subject |
| `patient_list_foldwise.txt` | Official train/test fold assignments |

**Diagnosis abbreviations** (non-exhaustive):

| Code | Condition |
|---|---|
| URTI | Upper Respiratory Tract Infection |
| LRTI | Lower Respiratory Tract Infection |
| COPD | Chronic Obstructive Pulmonary Disease |
| Bronchiectasis | Bronchiectasis |
| Pneumonia | Pneumonia |
| Asthma | Asthma |
| Healthy | No pathology |

---

## Sound Categories

Respiratory cycles are annotated into four categories by **expert clinicians**:

| Category | Description |
|---|---|
| **Normal** | No adventitious sounds |
| **Crackles** | Discontinuous, explosive sounds; associated with COPD, pneumonia, fibrosis |
| **Wheezes** | Continuous, high-pitched tonal sounds; associated with asthma, bronchospasm |
| **Crackles + Wheezes** | Both adventitious sounds present simultaneously |

---

## Recording Equipment

| Code | Device |
|---|---|
| `AKGC417L` | AKG C417L Microphone |
| `LittC2SE` | 3M Littmann Classic II SE Stethoscope |
| `Litt3200` | 3M Littmann 3200 Electronic Stethoscope |
| `Meditron` | WelchAllyn Meditron Master Elite Electronic Stethoscope |

---

## Chest Locations

Recordings were acquired from **seven** standardised chest positions:

| Code | Location |
|---|---|
| `Tc` | Trachea |
| `Al` | Anterior Left |
| `Ar` | Anterior Right |
| `Pl` | Posterior Left |
| `Pr` | Posterior Right |
| `Ll` | Lateral Left |
| `Lr` | Lateral Right |

---

## Train / Test Split

The official split divides subjects into a **60% training** and **40% test** set (approximately). Subject-level splitting ensures patient independence between sets.

An alternative **80/20** random split is also commonly used in the literature.

```
Official split  →  ~60% train / ~40% test   (patient_list_foldwise.txt, fold 4 = test)
Random split    →  ~80% train / ~20% test
```

> **Important:** Always perform subject-level splitting to avoid data leakage between respiratory cycles from the same patient.

---

## Evaluation Metrics

The ICBHI challenge uses the following metrics computed over the four sound classes:

| Metric | Formula |
|---|---|
| **Sensitivity (Se)** | Average per-class recall across all four categories |
| **Specificity (Sp)** | Average per-class true negative rate |
| **ICBHI Score** | `(Se + Sp) / 2` — the official competition metric |

Higher ICBHI Score → better overall classification performance.

---

## Citation

If you use this database in your research, please cite the following paper:

```bibtex
@article{rocha2019open,
  title     = {An open access database for the evaluation of respiratory sound classification algorithms},
  author    = {Rocha, Bruno M and Filos, Dimitris and Mendes, Lu{\'\i}s and Vogiatzis, Ioannis
               and Perantoni, Eleni and Kaimakamis, Evangelos and Natsiavas, Pantelis
               and Oliveira, Ana and J{\'a}come, Cristina and Marques, Alda and Paiva, Rui Pedro},
  journal   = {Physiological Measurement},
  volume    = {40},
  number    = {3},
  pages     = {035001},
  year      = {2019},
  publisher = {IOP Publishing}
}
```

---

## Contact

Questions, comments, and feedback on the dataset are welcomed at:

📧 **icbhi_challenge@med.auth.gr**

---

## License

The ICBHI 2017 Respiratory Sound Database is made **freely available for research purposes**. Please refer to the [official challenge page](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge) for the full terms of use.