# Respiratory Sound Similarity Report
*Generated 2026-04-28*

---

## Configuration

| Parameter | Value |
| --- | --- |
| Sample rate | 22050 Hz |
| ICBHI healthy max chunk duration | 5.0 s |
| Patient ill max chunk duration | 0.4 s |
| Algorithms | mfcc, chroma, embedding, temporal, md5 |

---

## Dataset Summary

| Dataset | Description | Chunks |
| --- | --- | --- |
| ICBHI healthy | crackle=0, wheeze=0 | 3,803 |
| Patient ill | ral / ronkus / acilma | 1,146 |

### Patient ill — Diagnosis Breakdown

| Diagnosis | Chunks |
| --- | --- |
| ral | 689 |
| ronkus | 313 |
| acilma | 144 |

---

## MD5 Hash — Exact / Re-encoded Duplicate Check

Exact duplicate chunks found: **0**

---

## Algorithm Results

### mfcc — MFCCs + cosine similarity — compare music style/timbre

Pairs: **4,358,238** (3,803 icbhi × 1,146 patient)

| Metric | Value |
| --- | --- |
| Mean | 0.8826 |
| Median | 0.9024 |
| Std | 0.0862 |
| Min | 0.2199 |
| Max | 0.9998 |

| Range | Pairs | Percentage |
| --- | --- | --- |
| [0.00, 0.25) | 9 | 0.00% |
| [0.25, 0.50) | 10,308 | 0.24% |
| [0.50, 0.75) | 343,034 | 7.87% |
| [0.75, 1.00) | 4,004,887 | 91.89% |

**Top 5 most similar**

| Similarity | ICBHI file | ICBHI window | Patient file | Patient window | Diagnosis |
| --- | --- | --- | --- | --- | --- |
| 0.9998 | 178_2b2_Al_m… | 10.52s–12.67s | ec6d893b2c9b… | 5.25s–5.32s | acilma |
| 0.9997 | 207_2b3_Tc_m… | 0.74s–4.27s | 778b440a0846… | 23.04s–23.10s | ral |
| 0.9997 | 204_2b5_Ar_m… | 18.52s–19.98s | 9c73dd357d95… | 10.57s–10.62s | ral |
| 0.9997 | 146_8p3_Pr_m… | 6.47s–8.91s | a385e19dcc03… | 7.41s–7.48s | ral |
| 0.9997 | 193_7b3_Ar_m… | 0.36s–1.89s | 9a45a0a0bd18… | 5.48s–5.55s | ral |

**Top 5 least similar**

| Similarity | ICBHI file | ICBHI window | Patient file | Patient window | Diagnosis |
| --- | --- | --- | --- | --- | --- |
| 0.2199 | 166_1p1_Al_s… | 26.70s–31.70s | c6c683f67127… | 7.19s–7.24s | ronkus |
| 0.2299 | 166_1p1_Al_s… | 21.70s–26.70s | c6c683f67127… | 7.19s–7.24s | ronkus |
| 0.2327 | 166_1p1_Al_s… | 26.70s–31.70s | cfe363d963af… | 23.44s–23.84s | ral |
| 0.2340 | 166_1p1_Al_s… | 26.70s–31.70s | b2262522fcfd… | 10.32s–10.37s | ral |
| 0.2416 | 166_1p1_Al_s… | 21.70s–26.70s | cfe363d963af… | 23.44s–23.84s | ral |

---

### chroma — Acoustic fingerprinting — chroma features

Pairs: **4,358,238** (3,803 icbhi × 1,146 patient)

| Metric | Value |
| --- | --- |
| Mean | 0.9087 |
| Median | 0.9304 |
| Std | 0.0745 |
| Min | 0.2896 |
| Max | 0.9993 |

| Range | Pairs | Percentage |
| --- | --- | --- |
| [0.00, 0.25) | 0 | 0.00% |
| [0.25, 0.50) | 2,421 | 0.06% |
| [0.50, 0.75) | 201,048 | 4.61% |
| [0.75, 1.00) | 4,154,769 | 95.33% |

**Top 5 most similar**

| Similarity | ICBHI file | ICBHI window | Patient file | Patient window | Diagnosis |
| --- | --- | --- | --- | --- | --- |
| 0.9993 | 110_1b1_Pr_s… | 36.82s–36.84s | b5403d11eadd… | 2.58s–2.58s | ral |
| 0.9989 | 110_1b1_Pr_s… | 36.82s–36.84s | 8612fa3c1892… | 18.48s–18.48s | acilma |
| 0.9988 | 170_1b3_Tc_m… | 13.13s–16.30s | 8d5841abf718… | 1.72s–2.12s | ral |
| 0.9987 | 110_1b1_Pr_s… | 36.82s–36.84s | 5af8ab7354c3… | 19.74s–19.74s | ral |
| 0.9987 | 170_1b3_Pr_m… | 10.12s–13.13s | 8d5841abf718… | 1.72s–2.12s | ral |

**Top 5 least similar**

| Similarity | ICBHI file | ICBHI window | Patient file | Patient window | Diagnosis |
| --- | --- | --- | --- | --- | --- |
| 0.2896 | 157_1b1_Lr_s… | 61.75s–66.50s | b5403d11eadd… | 2.57s–2.97s | acilma |
| 0.2954 | 157_1b1_Pl_s… | 10.13s–10.97s | b5403d11eadd… | 2.57s–2.97s | acilma |
| 0.2969 | 157_1b1_Pl_s… | 23.75s–24.15s | b5403d11eadd… | 2.57s–2.97s | acilma |
| 0.2982 | 118_1b1_Pr_s… | 11.38s–12.46s | b5403d11eadd… | 2.57s–2.97s | acilma |
| 0.2985 | 193_1b2_Ll_m… | 4.52s–6.23s | f7704e06a5e7… | 2.63s–2.95s | acilma |

---

### embedding — ML embeddings — semantic/mood similarity

Pairs: **4,358,238** (3,803 icbhi × 1,146 patient)

| Metric | Value |
| --- | --- |
| Mean | 0.9495 |
| Median | 0.9546 |
| Std | 0.0275 |
| Min | 0.6046 |
| Max | 0.9982 |

| Range | Pairs | Percentage |
| --- | --- | --- |
| [0.00, 0.25) | 0 | 0.00% |
| [0.25, 0.50) | 0 | 0.00% |
| [0.50, 0.75) | 1,392 | 0.03% |
| [0.75, 1.00) | 4,356,846 | 99.97% |

**Top 5 most similar**

| Similarity | ICBHI file | ICBHI window | Patient file | Patient window | Diagnosis |
| --- | --- | --- | --- | --- | --- |
| 0.9982 | 135_2b3_Al_m… | 19.64s–19.96s | 73542646cd5b… | 11.94s–12.34s | ral |
| 0.9980 | 110_1b1_Pr_s… | 36.82s–36.84s | 5af8ab7354c3… | 4.59s–4.60s | ral |
| 0.9980 | 135_2b3_Pl_m… | 19.64s–19.98s | 2c5f90f58c06… | 5.25s–5.55s | ral |
| 0.9979 | 221_2b2_Pl_m… | 19.51s–19.99s | 73542646cd5b… | 11.94s–12.34s | ral |
| 0.9979 | 135_2b3_Pl_m… | 19.64s–19.98s | 4bc58064deb8… | 4.12s–4.52s | ral |

**Top 5 least similar**

| Similarity | ICBHI file | ICBHI window | Patient file | Patient window | Diagnosis |
| --- | --- | --- | --- | --- | --- |
| 0.6046 | 151_3p2_Tc_m… | 0.06s–3.51s | 8612fa3c1892… | 10.34s–10.34s | acilma |
| 0.6151 | 151_3p2_Tc_m… | 0.06s–3.51s | 5af8ab7354c3… | 18.13s–18.14s | ral |
| 0.6156 | 151_3p2_Tc_m… | 0.06s–3.51s | 8612fa3c1892… | 18.48s–18.48s | acilma |
| 0.6186 | 151_3p2_Tc_m… | 0.06s–3.51s | 5af8ab7354c3… | 19.74s–19.74s | ral |
| 0.6205 | 151_3p2_Tc_m… | 0.06s–3.51s | 8612fa3c1892… | 9.10s–9.10s | ronkus |

---

### temporal — Cross-correlation proxy — detect time-shifted copies

Pairs: **4,358,238** (3,803 icbhi × 1,146 patient)

| Metric | Value |
| --- | --- |
| Mean | 0.7745 |
| Median | 0.8120 |
| Std | 0.1378 |
| Min | -0.0653 |
| Max | 0.9947 |

| Range | Pairs | Percentage |
| --- | --- | --- |
| [0.00, 0.25) | 21,005 | 0.48% |
| [0.25, 0.50) | 216,286 | 4.96% |
| [0.50, 0.75) | 1,172,370 | 26.90% |
| [0.75, 1.00) | 2,948,549 | 67.65% |

**Top 5 most similar**

| Similarity | ICBHI file | ICBHI window | Patient file | Patient window | Diagnosis |
| --- | --- | --- | --- | --- | --- |
| 0.9947 | 130_3b4_Ar_m… | 5.92s–9.96s | 74473b6c4cf6… | 6.60s–6.60s | ral |
| 0.9938 | 185_1b1_Ll_s… | 0.63s–2.43s | 8612fa3c1892… | 18.48s–18.48s | acilma |
| 0.9938 | 185_1b1_Ll_s… | 0.63s–2.43s | 8612fa3c1892… | 10.34s–10.34s | acilma |
| 0.9938 | 185_1b1_Ll_s… | 0.63s–2.43s | 8612fa3c1892… | 9.10s–9.10s | ronkus |
| 0.9938 | 185_1b1_Ll_s… | 0.63s–2.43s | b5403d11eadd… | 2.58s–2.58s | ral |

**Top 5 least similar**

| Similarity | ICBHI file | ICBHI window | Patient file | Patient window | Diagnosis |
| --- | --- | --- | --- | --- | --- |
| -0.0653 | 118_1b1_Pr_s… | 6.38s–11.38s | c6c683f67127… | 30.29s–30.61s | ronkus |
| -0.0629 | 104_1b1_Pr_s… | 15.01s–18.44s | d3feb6e913da… | 17.78s–17.98s | ronkus |
| -0.0492 | 118_1b1_Pr_s… | 6.38s–11.38s | 2c5f90f58c06… | 25.90s–25.99s | ronkus |
| -0.0466 | 118_1b1_Pr_s… | 6.38s–11.38s | a385e19dcc03… | 28.24s–28.30s | ral |
| -0.0319 | 118_1b1_Pr_s… | 6.38s–11.38s | 23b2f1d08bc1… | 9.57s–9.87s | ronkus |

---

*Report generated by `report.py`.*
