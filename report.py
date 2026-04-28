import os
from collections import Counter
from datetime import date

import numpy as np
import pandas as pd
from tqdm import tqdm

from calculation import SAMPLE_RATE, MAX_ICBHI_DURATION, MAX_PATIENT_DURATION

REPORT_PATH = "report.md"
CSV_PATH = "report.csv"

DIST_BUCKETS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
TOP_N = 5


def _table(headers: list[str], rows: list[list]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    sep_row    = "| " + " | ".join("---" for _ in headers) + " |"
    data_rows  = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join([header_row, sep_row] + data_rows)


def _build_report(sim: np.ndarray, icbhi_meta: list[dict], patient_meta: list[dict]) -> str:
    n_icbhi, n_patient = sim.shape
    avg_icbhi   = np.mean(sim, axis=1)
    avg_patient = np.mean(sim, axis=0)

    diag_counts = Counter(m["diagnosis"] for m in patient_meta)

    flat = sim.ravel()
    dist_rows = []
    for lo, hi in DIST_BUCKETS:
        count = int(np.sum((flat >= lo) & (flat < hi)))
        dist_rows.append([f"[{lo:.2f}, {hi:.2f})", f"{count:,}", f"{count / flat.size * 100:.2f}%"])

    flat_desc = np.argsort(flat)[::-1]
    flat_asc  = flat_desc[::-1]

    def pair_rows(flat_indices: np.ndarray, n: int) -> list[list]:
        rows = []
        for fi in tqdm(flat_indices[:n], desc="Ranking pairs", leave=False, unit="pair"):
            i, j = np.unravel_index(fi, sim.shape)
            icbhi_m, patient_m = icbhi_meta[i], patient_meta[j]
            rows.append([
                f"{sim[i, j]:.4f}",
                icbhi_m["file"][:12] + "…",
                f"{icbhi_m['start']:.2f}s – {icbhi_m['end']:.2f}s",
                patient_m["file"][:12] + "…",
                f"{patient_m['start']:.2f}s – {patient_m['end']:.2f}s",
                patient_m.get("diagnosis", "—"),
            ])
        return rows

    pair_headers = ["Similarity", "ICBHI file", "ICBHI window", "Patient file", "Patient window", "Diagnosis"]

    lines = [
        "# Respiratory Sound Similarity Report",
        f"*Generated {date.today().isoformat()}*",
        "",
        "---",
        "",
        "## Configuration",
        "",
        _table(
            ["Parameter", "Value"],
            [
                ["Sample rate", f"{SAMPLE_RATE} Hz"],
                ["ICBHI healthy max chunk duration", f"{MAX_ICBHI_DURATION} s"],
                ["Patient ill max chunk duration", f"{MAX_PATIENT_DURATION} s"],
                ["Feature vector length", "30 (13 MFCC mean + 13 MFCC std + 4 spectral)"],
                ["Similarity metric", "Cosine similarity"],
            ],
        ),
        "",
        "---",
        "",
        "## Dataset Summary",
        "",
        _table(
            ["Dataset", "Description", "Chunks"],
            [
                ["ICBHI healthy", "crackle=0, wheeze=0", f"{n_icbhi:,}"],
                ["Patient ill",   "ral / ronkus / acilma", f"{n_patient:,}"],
            ],
        ),
        "",
        "### Patient ill — Diagnosis Breakdown",
        "",
        _table(
            ["Diagnosis", "Chunks"],
            [[label, f"{cnt:,}"] for label, cnt in sorted(diag_counts.items(), key=lambda x: -x[1])],
        ),
        "",
        "---",
        "",
        "## Overall Similarity Statistics",
        "",
        f"Total pairs compared: **{sim.size:,}** ({n_icbhi:,} icbhi × {n_patient:,} patient)",
        "",
        _table(
            ["Metric", "Value"],
            [
                ["Mean",   f"{np.mean(sim):.4f}"],
                ["Median", f"{np.median(sim):.4f}"],
                ["Std",    f"{np.std(sim):.4f}"],
                ["Min",    f"{np.min(sim):.4f}"],
                ["Max",    f"{np.max(sim):.4f}"],
            ],
        ),
        "",
        "---",
        "",
        "## Per-Chunk Average Similarity",
        "",
        "### ICBHI healthy — avg similarity to all patient ill chunks",
        "",
        _table(
            ["Metric", "Value"],
            [
                ["Mean",   f"{np.mean(avg_icbhi):.4f}"],
                ["Median", f"{np.median(avg_icbhi):.4f}"],
                ["Min",    f"{np.min(avg_icbhi):.4f}"],
                ["Max",    f"{np.max(avg_icbhi):.4f}"],
            ],
        ),
        "",
        "### Patient ill — avg similarity to all ICBHI healthy chunks",
        "",
        _table(
            ["Metric", "Value"],
            [
                ["Mean",   f"{np.mean(avg_patient):.4f}"],
                ["Median", f"{np.median(avg_patient):.4f}"],
                ["Min",    f"{np.min(avg_patient):.4f}"],
                ["Max",    f"{np.max(avg_patient):.4f}"],
            ],
        ),
        "",
        "---",
        "",
        "## Similarity Distribution",
        "",
        _table(["Range", "Pairs", "Percentage"], dist_rows),
        "",
        "---",
        "",
        f"## Top {TOP_N} Most Similar Pairs",
        "",
        _table(pair_headers, pair_rows(flat_desc, TOP_N)),
        "",
        f"## Top {TOP_N} Least Similar Pairs",
        "",
        _table(pair_headers, pair_rows(flat_asc, TOP_N)),
        "",
        "---",
        "",
        "*Report generated by `report.py`.*",
    ]

    return "\n".join(lines) + "\n"


def _write_csv(sim: np.ndarray, icbhi_meta: list[dict], patient_meta: list[dict]) -> None:
    n_icbhi, n_patient = sim.shape
    ii, pi = np.meshgrid(np.arange(n_icbhi), np.arange(n_patient), indexing="ij")
    ii, pi = ii.ravel(), pi.ravel()

    rows = {
        "icbhi_file":       [icbhi_meta[i]["file"]            for i in tqdm(ii, desc="Building CSV", unit="pair")],
        "icbhi_start":      [icbhi_meta[i]["start"]           for i in ii],
        "icbhi_end":        [icbhi_meta[i]["end"]             for i in ii],
        "patient_file":     [patient_meta[j]["file"]          for j in pi],
        "patient_start":    [patient_meta[j]["start"]         for j in pi],
        "patient_end":      [patient_meta[j]["end"]           for j in pi],
        "patient_diagnosis":[patient_meta[j].get("diagnosis", "") for j in pi],
        "similarity":       sim.ravel(),
    }

    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
    print(f"  Saved → {CSV_PATH}")


def generate_report(sim: np.ndarray, icbhi_meta: list[dict], patient_meta: list[dict]) -> None:
    """Write report.md and report.csv from an already-computed similarity matrix and metadata."""
    print("=== Writing report ===")
    text = _build_report(sim, icbhi_meta, patient_meta)
    with open(REPORT_PATH, "w") as f:
        f.write(text)
    print(f"  Saved → {REPORT_PATH}")
    _write_csv(sim, icbhi_meta, patient_meta)
