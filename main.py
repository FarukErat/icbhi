import os
import glob

import numpy as np
import librosa
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from calculation import extract_features, SAMPLE_RATE, MAX_ICBHI_DURATION, MAX_PATIENT_DURATION
from report import generate_report

ICBHI_DIR = "ICBHI_final_database"
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


def load_icbhi_healthy() -> tuple[np.ndarray, list[dict]]:
    """
    ICBHI segments where crackles=0 and wheezes=0, chopped to MAX_ICBHI_DURATION.
    Returns (feature_matrix, metadata_list).
    metadata keys: file, start, end
    """
    txt_files = glob.glob(os.path.join(ICBHI_DIR, "*.txt"))
    features: list[np.ndarray] = []
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
                features.append(extract_features(chunk, sr))
                meta.append({"file": file_id, "start": seg_start + c_start, "end": seg_start + c_end})

    return np.array(features), meta


def load_patient_ill() -> tuple[np.ndarray, list[dict]]:
    """
    Patient segments whose diagnosis is not 'normal', chopped to MAX_PATIENT_DURATION.
    Segments from both doctors are deduplicated by (start, end).
    Returns (feature_matrix, metadata_list).
    metadata keys: file, start, end, diagnosis
    """
    wav_files = glob.glob(os.path.join(PATIENTS_DIR, "*.wav"))
    features: list[np.ndarray] = []
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
                features.append(extract_features(chunk, sr))
                meta.append({"file": file_id, "start": seg_start + c_start, "end": seg_start + c_end, "diagnosis": diagnosis})

    return np.array(features), meta


def main() -> None:
    print("=== Loading ICBHI healthy segments ===")
    icbhi_features, icbhi_meta = load_icbhi_healthy()
    print(f"  {len(icbhi_features)} feature vectors\n")

    print("=== Loading patient ill segments ===")
    patient_features, patient_meta = load_patient_ill()
    print(f"  {len(patient_features)} feature vectors\n")

    if len(icbhi_features) == 0 or len(patient_features) == 0:
        print("ERROR: one of the sets is empty — nothing to compare.")
        return

    print("=== Computing pairwise cosine similarity ===")
    sim = cosine_similarity(icbhi_features, patient_features)

    print(f"\n  Pairs compared : {sim.size:,}  ({len(icbhi_features)} icbhi × {len(patient_features)} patient)")
    print(f"  Mean           : {np.mean(sim):.4f}")
    print(f"  Median         : {np.median(sim):.4f}")
    print(f"  Std            : {np.std(sim):.4f}")
    print(f"  Min            : {np.min(sim):.4f}")
    print(f"  Max            : {np.max(sim):.4f}")

    generate_report(sim, icbhi_meta, patient_meta)


if __name__ == "__main__":
    main()
