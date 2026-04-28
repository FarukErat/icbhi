import os
import glob

import numpy as np
import librosa
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from calculation import (
    extract_all_features, md5_fingerprint,
    SAMPLE_RATE, MAX_ICBHI_DURATION, MAX_PATIENT_DURATION, ALGORITHMS,
)
from report import generate_report

ICBHI_DIR    = "ICBHI_final_database"
PATIENTS_DIR = "patients"
DIAGNOSES_DIR = os.path.join(PATIENTS_DIR, "diagnoses")


def split_audio(audio: np.ndarray, sr: int, max_duration: float) -> list[tuple[np.ndarray, float, float]]:
    """Chop audio into back-to-back chunks; yields (chunk, start_sec, end_sec) within the segment."""
    max_samples = int(max_duration * sr)
    result = []
    for i in range(0, len(audio), max_samples):
        chunk = audio[i:i + max_samples]
        if len(chunk) > 0:
            result.append((chunk, i / sr, (i + len(chunk)) / sr))
    return result


def load_icbhi_healthy() -> tuple[dict[str, np.ndarray], list[dict]]:
    """
    ICBHI segments where crackles=0 and wheezes=0, chopped to MAX_ICBHI_DURATION.
    Returns (features_by_algorithm, metadata_list).
    metadata keys: file, start, end, md5
    """
    txt_files = glob.glob(os.path.join(ICBHI_DIR, "*.txt"))
    features: dict[str, list] = {algo: [] for algo in ALGORITHMS}
    meta: list[dict] = []

    for txt_path in tqdm(txt_files, desc="ICBHI healthy", unit="file"):
        wav_path = txt_path.replace(".txt", ".wav")
        if not os.path.exists(wav_path):
            continue

        healthy_slices: list[tuple[float, float]] = []
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4 and int(parts[2]) == 0 and int(parts[3]) == 0:
                    healthy_slices.append((float(parts[0]), float(parts[1])))

        if not healthy_slices:
            continue

        try:
            audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
        except Exception as e:
            tqdm.write(f"[skip] {wav_path}: {e}")
            continue

        file_id = os.path.basename(wav_path).replace(".wav", "")
        for seg_start, seg_end in healthy_slices:
            segment = audio[int(seg_start * sr):int(seg_end * sr)]
            for chunk, c_start, c_end in split_audio(segment, sr, MAX_ICBHI_DURATION):
                chunk_features = extract_all_features(chunk, sr)
                for algo in ALGORITHMS:
                    features[algo].append(chunk_features[algo])
                meta.append({
                    "file":  file_id,
                    "start": seg_start + c_start,
                    "end":   seg_start + c_end,
                    "md5":   md5_fingerprint(chunk),
                })

    return {algo: np.array(vecs) for algo, vecs in features.items()}, meta


def load_patient_ill() -> tuple[dict[str, np.ndarray], list[dict]]:
    """
    Patient segments whose diagnosis is not 'normal', chopped to MAX_PATIENT_DURATION.
    Segments from both doctors are deduplicated by (start, end).
    Returns (features_by_algorithm, metadata_list).
    metadata keys: file, start, end, diagnosis, md5
    """
    wav_files = glob.glob(os.path.join(PATIENTS_DIR, "*.wav"))
    features: dict[str, list] = {algo: [] for algo in ALGORITHMS}
    meta: list[dict] = []

    for wav_path in tqdm(wav_files, desc="Patient ill", unit="file"):
        file_id = os.path.basename(wav_path).replace(".wav", "")

        ill_slices: list[tuple[float, float, str]] = []
        seen: set[tuple[float, float]] = set()

        for doctor in ("Fatih", "Guney"):
            csv_path = os.path.join(DIAGNOSES_DIR, doctor, f"{file_id}.csv")
            if not os.path.exists(csv_path):
                continue
            try:
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    if str(row["diagnosis"]).strip().lower() == "normal":
                        continue
                    key = (round(float(row["start"]), 3), round(float(row["end"]), 3))
                    if key not in seen:
                        seen.add(key)
                        ill_slices.append((float(row["start"]), float(row["end"]), str(row["diagnosis"]).strip()))
            except Exception as e:
                tqdm.write(f"[skip] {csv_path}: {e}")

        if not ill_slices:
            continue

        try:
            audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
        except Exception as e:
            tqdm.write(f"[skip] {wav_path}: {e}")
            continue

        for seg_start, seg_end, diagnosis in ill_slices:
            segment = audio[int(seg_start * sr):int(seg_end * sr)]
            for chunk, c_start, c_end in split_audio(segment, sr, MAX_PATIENT_DURATION):
                chunk_features = extract_all_features(chunk, sr)
                for algo in ALGORITHMS:
                    features[algo].append(chunk_features[algo])
                meta.append({
                    "file":      file_id,
                    "start":     seg_start + c_start,
                    "end":       seg_start + c_end,
                    "diagnosis": diagnosis,
                    "md5":       md5_fingerprint(chunk),
                })

    return {algo: np.array(vecs) for algo, vecs in features.items()}, meta


def main() -> None:
    print("=== Loading ICBHI healthy segments ===")
    icbhi_features, icbhi_meta = load_icbhi_healthy()
    print(f"  {len(icbhi_meta)} chunks\n")

    print("=== Loading patient ill segments ===")
    patient_features, patient_meta = load_patient_ill()
    print(f"  {len(patient_meta)} chunks\n")

    if len(icbhi_meta) == 0 or len(patient_meta) == 0:
        print("ERROR: one of the sets is empty — nothing to compare.")
        return

    print("=== Computing pairwise cosine similarity per algorithm ===")
    sim_matrices: dict[str, np.ndarray] = {}
    for algo in ALGORITHMS:
        sim_matrices[algo] = cosine_similarity(icbhi_features[algo], patient_features[algo])
        m = sim_matrices[algo]
        print(f"  [{algo}]  mean={np.mean(m):.4f}  median={np.median(m):.4f}  "
              f"std={np.std(m):.4f}  min={np.min(m):.4f}  max={np.max(m):.4f}")

    icbhi_hashes   = {m["md5"] for m in icbhi_meta}
    patient_hashes = {m["md5"] for m in patient_meta}
    md5_matches = len(icbhi_hashes & patient_hashes)
    print(f"\n  [md5]  exact duplicate chunks: {md5_matches}")

    generate_report(sim_matrices, icbhi_meta, patient_meta, md5_matches)


if __name__ == "__main__":
    main()
