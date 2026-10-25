"""
Production Training Pipeline for Autonomic Nervous System Fatigue Classification.

Trains, validates, and exports a production-grade machine learning model using
real physiological data from the WESAD (Wearable Stress and Affect Detection) dataset.

Pipeline Architecture:
1. Multi-Subject Data Ingestion (WESAD subjects S2 - S17).
2. SciPy DSP 3rd-order Butterworth bandpass filtering (0.75 - 3.0 Hz / 45 - 180 BPM).
3. Systolic peak detection and physiological outlier RR interval rejection (330 - 1500 ms).
4. Sliding Window Feature Engineering (45s window, 15s stride):
   - Time-Domain: rMSSD, SDNN, HR Mean, HR Std, pNN50, pNN20, CV_RR
   - Non-Linear Poincare Geometry: SD1, SD2, SD1/SD2 ratio
   - Autonomic Index: Baevsky's Stress Index (SI)
   - Frequency-Domain (Welch PSD): LF Power, HF Power, LF/HF Ratio, Total Power
   - Optical Morphometrics: Perfusion Index, Pulse Rise Time
5. GroupKFold / LOSO Cross-Validation (prevents subject data leakage).
6. Model Tournament: Random Forest vs XGBoost Classifier.
7. Metrics Logging & Feature Importance Analysis.
8. Artifact Export (Joblib bundle, Scaler JSON, ONNX model).
"""

import os
import sys
import glob
import time
import json
import pickle
import argparse
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
from scipy import signal, interpolate, integrate
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)
import joblib

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# Canonical 18-Feature Vector matching production inference engine
FEATURE_NAMES = [
    "hr_mean",
    "hr_std",
    "rr_mean",
    "sdnn",
    "rmssd",
    "pnn50",
    "pnn20",
    "cv_rr",
    "sd1",
    "sd2",
    "sd1_sd2_ratio",
    "stress_index",
    "lf_power",
    "hf_power",
    "lf_hf_ratio",
    "total_power",
    "perfusion_index",
    "pulse_rise_time_ms",
]

CLASS_NAMES = ["GREEN", "YELLOW", "RED"]
CLASS_DESCRIPTIONS = {
    0: "GREEN (Rested / High Parasympathetic Tone / Baseline / Recovery)",
    1: "YELLOW (Moderate Strain / Transient Sympathetic Shift / Amusement)",
    2: "RED (Acute Sympathetic Activation / Deep Fatigue / TSST Stress)",
}


def butterworth_bandpass_filter(
    data: np.ndarray,
    fs: float,
    lowcut: float = 0.75,
    highcut: float = 3.0,
    order: int = 3,
) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass filter to isolate human cardiac pulses.
    
    :param data: Input raw 1D signal.
    :param fs: Sampling rate in Hz.
    :param lowcut: Low cutoff in Hz (0.75 Hz = 45 BPM).
    :param highcut: High cutoff in Hz (3.0 Hz = 180 BPM).
    :param order: Filter order (3rd order).
    :return: Filtered 1D array.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.95)
    b, a = signal.butter(order, [low, high], btype="bandpass")
    detrended = signal.detrend(data)
    filtered = signal.filtfilt(b, a, detrended)
    return filtered


def extract_window_biometrics(
    raw_window: np.ndarray,
    fs: float = 64.0,
) -> Optional[Dict[str, float]]:
    """
    Extract full 18-feature biometric matrix from a single 45-second signal window.
    """
    if len(raw_window) < int(10 * fs):
        return None

    # 1. Digital Bandpass Filter
    filtered = butterworth_bandpass_filter(raw_window, fs=fs, lowcut=0.75, highcut=3.0, order=3)

    # 2. Systolic Peak Detection with physiological refractory period
    min_distance = max(int(0.33 * fs), 5)  # Max HR ~180 BPM
    sig_std = np.std(filtered)
    if sig_std < 1e-4:
        return None
    prominence = max(0.12 * sig_std, 1e-4)

    peaks, _ = signal.find_peaks(filtered, distance=min_distance, prominence=prominence)
    if len(peaks) < 8:
        return None

    # Peak timings and raw RR intervals
    peak_times_ms = peaks * (1000.0 / fs)
    raw_rr = np.diff(peak_times_ms)

    # 3. Physiological Outlier Rejection (330 ms to 1500 ms / 40 - 180 BPM)
    valid_bounds = (raw_rr >= 330.0) & (raw_rr <= 1500.0)
    clean_rr = raw_rr[valid_bounds]
    if len(clean_rr) < 6:
        return None

    # Local median jump filter (removes sudden ectopic step deviations > 35%)
    med_rr = np.median(clean_rr)
    clean_rr = clean_rr[np.abs(clean_rr - med_rr) <= 0.35 * med_rr]
    if len(clean_rr) < 5:
        return None

    # 4. Time-Domain HRV Metrics
    mean_rr = float(np.mean(clean_rr))
    hr_bpm = 60000.0 / mean_rr
    hr_std = float(np.std(60000.0 / clean_rr))
    sdnn = float(np.std(clean_rr))
    rr_diffs = np.diff(clean_rr)
    rmssd = float(np.sqrt(np.mean(rr_diffs ** 2)))
    pnn50 = float(np.sum(np.abs(rr_diffs) > 50.0) / len(rr_diffs) * 100.0) if len(rr_diffs) > 0 else 0.0
    pnn20 = float(np.sum(np.abs(rr_diffs) > 20.0) / len(rr_diffs) * 100.0) if len(rr_diffs) > 0 else 0.0
    cv_rr = float(sdnn / (mean_rr + 1e-6))

    # 5. Non-Linear Poincaré Geometry
    sd1 = float(np.sqrt(0.5 * (rmssd ** 2)))
    sd2 = float(np.sqrt(max(1e-6, 2.0 * (sdnn ** 2) - 0.5 * (rmssd ** 2))))
    sd1_sd2_ratio = float(sd1 / (sd2 + 1e-6))

    # 6. Baevsky's Stress Index (SI)
    bin_width = 50.0
    counts, bin_edges = np.histogram(clean_rr, bins=np.arange(300, 1600, bin_width))
    mode_idx = np.argmax(counts)
    mo = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2.0 / 1000.0
    amo = (counts[mode_idx] / len(clean_rr)) * 100.0
    mxdmn = (np.max(clean_rr) - np.min(clean_rr)) / 1000.0
    stress_index = float(amo / (2.0 * (mo + 1e-6) * (mxdmn + 1e-6)))

    # 7. Frequency-Domain (Welch PSD on 4Hz Resampled RR series)
    rr_cum_times = np.cumsum(clean_rr) / 1000.0
    rr_cum_times = rr_cum_times - rr_cum_times[0]
    total_window_span = rr_cum_times[-1]

    if total_window_span > 10.0:
        t_interp = np.arange(0, total_window_span, 0.25)  # 4 Hz uniform grid
        interp_fn = interpolate.interp1d(rr_cum_times, clean_rr, kind="linear", fill_value="extrapolate")
        rr_resampled = interp_fn(t_interp)

        nperseg = min(len(rr_resampled), 64)
        f_welch, psd = signal.welch(rr_resampled, fs=4.0, nperseg=nperseg)
        lf_mask = (f_welch >= 0.04) & (f_welch < 0.15)
        hf_mask = (f_welch >= 0.15) & (f_welch <= 0.40)

        # Cross-version trapezoidal integration (NumPy 2.x & 1.x compatible)
        trapz_fn = getattr(integrate, "trapezoid", getattr(np, "trapz", None))
        lf_power = float(trapz_fn(psd[lf_mask], f_welch[lf_mask])) if np.any(lf_mask) else 0.0
        hf_power = float(trapz_fn(psd[hf_mask], f_welch[hf_mask])) if np.any(hf_mask) else 0.0
        lf_hf_ratio = float(lf_power / (hf_power + 1e-6))
        total_power = float(lf_power + hf_power)
    else:
        lf_power, hf_power, lf_hf_ratio, total_power = 0.0, 0.0, 1.0, 0.0

    # 8. Optical Morphometrics (Perfusion Index & Pulse Rise Time)
    dc_level = float(np.mean(raw_window))
    ac_amp = float(np.percentile(filtered, 95) - np.percentile(filtered, 5))
    perfusion_index = float((ac_amp / (abs(dc_level) + 1e-6)) * 100.0)

    rise_times = []
    for i in range(len(peaks)):
        p = peaks[i]
        lookback = int(0.3 * fs)
        seg = filtered[max(0, p - lookback) : p]
        if len(seg) > 2:
            v_idx = int(np.argmin(seg))
            rise_ms = (len(seg) - v_idx) * (1000.0 / fs)
            if 40.0 <= rise_ms <= 350.0:
                rise_times.append(rise_ms)
    pulse_rise_time = float(np.mean(rise_times)) if len(rise_times) > 0 else 115.0

    return {
        "hr_mean": round(hr_bpm, 2),
        "hr_std": round(hr_std, 2),
        "rr_mean": round(mean_rr, 2),
        "sdnn": round(sdnn, 2),
        "rmssd": round(rmssd, 2),
        "pnn50": round(pnn50, 2),
        "pnn20": round(pnn20, 2),
        "cv_rr": round(cv_rr, 4),
        "sd1": round(sd1, 2),
        "sd2": round(sd2, 2),
        "sd1_sd2_ratio": round(sd1_sd2_ratio, 4),
        "stress_index": round(stress_index, 2),
        "lf_power": round(lf_power, 2),
        "hf_power": round(hf_power, 2),
        "lf_hf_ratio": round(lf_hf_ratio, 4),
        "total_power": round(total_power, 2),
        "perfusion_index": round(perfusion_index, 3),
        "pulse_rise_time_ms": round(pulse_rise_time, 2),
    }


def parse_wesad_subject(
    pkl_path: str,
    modality: str = "bvp",
    window_sec: float = 45.0,
    step_sec: float = 15.0,
) -> List[Dict[str, Any]]:
    """
    Extract sliding windows from a single subject's pickle file.
    
    Condition Mapping:
      - WESAD 1 (Baseline) & 4 (Meditation) -> Class 0 (GREEN: Rested / Recovery)
      - WESAD 3 (Amusement) -> Class 1 (YELLOW: Moderate Strain / Mild Depletion)
      - WESAD 2 (TSST Stress) -> Class 2 (RED: Acute Sympathetic Stress / Exhaustion)
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    subj_id = data.get("subject", os.path.basename(os.path.dirname(pkl_path)))
    labels_700hz = data["label"].flatten()

    if modality == "bvp":
        sig_data = data["signal"]["wrist"]["BVP"].flatten()
        fs_signal = 64.0
    else:
        sig_data = data["signal"]["chest"]["ECG"].flatten()
        fs_signal = 700.0

    fs_label = 700.0
    duration_sec = len(sig_data) / fs_signal
    extracted_rows: List[Dict[str, Any]] = []

    for start_t in np.arange(0, duration_sec - window_sec, step_sec):
        end_t = start_t + window_sec

        # Align ground truth labels (at 700 Hz)
        lbl_start = int(start_t * fs_label)
        lbl_end = int(end_t * fs_label)
        win_labels = labels_700hz[lbl_start:lbl_end]
        if len(win_labels) == 0:
            continue

        bincounts = np.bincount(win_labels)
        maj_label = int(bincounts.argmax())
        majority_ratio = bincounts[maj_label] / len(win_labels)

        # Enforce >= 70% label purity (discards transition periods)
        if majority_ratio < 0.70:
            continue

        # Map to 3-tier target classes
        if maj_label in (1, 4):
            target_class = 0  # Green (Baseline / Meditation)
        elif maj_label == 3:
            target_class = 1  # Yellow (Amusement / Mild Strain)
        elif maj_label == 2:
            target_class = 2  # Red (Stress / Exhaustion)
        else:
            continue  # Ignore transient/undefined labels 0, 5, 6, 7

        # Extract signal slice
        sig_start = int(start_t * fs_signal)
        sig_end = int(end_t * fs_signal)
        raw_slice = sig_data[sig_start:sig_end]

        # Extract features
        feats = extract_window_biometrics(raw_slice, fs=fs_signal)
        if feats is not None:
            feats["subject"] = subj_id
            feats["target"] = target_class
            feats["window_start_sec"] = round(start_t, 1)
            extracted_rows.append(feats)

    return extracted_rows


def load_or_extract_dataset(
    data_dir: str,
    cache_path: str,
    modality: str = "bvp",
    window_sec: float = 45.0,
    step_sec: float = 15.0,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Load extracted feature matrix from disk cache, or compute from raw WESAD pickle files.
    """
    if not os.path.exists(cache_path):
        backend_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), cache_path)
        if os.path.exists(backend_cache):
            cache_path = backend_cache

    if os.path.exists(cache_path) and not force_recompute:
        print(f"[Data Ingestion] Found cached feature dataset at {cache_path}. Loading...")
        df = pd.read_csv(cache_path)
        print(f"[Data Ingestion] Successfully loaded {len(df)} samples across {df['subject'].nunique()} subjects.")
        return df

    # Find subject directories
    sub_dirs = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("S")
    ]
    sub_dirs.sort(key=lambda s: int(s[1:]) if s[1:].isdigit() else s)

    if not sub_dirs:
        raise FileNotFoundError(f"No subject folders (S2, S3, ...) found inside '{data_dir}'.")

    print(f"[Data Ingestion] Discovered {len(sub_dirs)} WESAD subjects: {sub_dirs}")
    print(f"[Data Ingestion] Extracting {modality.upper()} biometrics (Window={window_sec}s, Stride={step_sec}s)...")

    all_rows = []
    t_start = time.time()
    for idx, subj in enumerate(sub_dirs, 1):
        pkl_path = os.path.join(data_dir, subj, f"{subj}.pkl")
        if not os.path.exists(pkl_path):
            print(f"  [Warning] Missing {pkl_path}, skipping.")
            continue

        subj_t0 = time.time()
        print(f"  [{idx}/{len(sub_dirs)}] Processing {subj}...", end="", flush=True)
        rows = parse_wesad_subject(pkl_path, modality=modality, window_sec=window_sec, step_sec=step_sec)
        all_rows.extend(rows)
        print(f" Done ({len(rows)} windows, {time.time() - subj_t0:.1f}s)")

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"[Data Ingestion] Extracted {len(df)} total windows across {df['subject'].nunique()} subjects in {time.time() - t_start:.1f}s.")
    print(f"[Data Ingestion] Cached dataset saved to {cache_path}")
    return df


def evaluate_model_cv(
    model_name: str,
    create_model_fn,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """
    Perform GroupKFold cross-validation to strictly prevent subject data leakage.
    Scalers are fit strictly on training splits.
    """
    gkf = GroupKFold(n_splits=n_splits)

    fold_accuracies = []
    fold_macro_f1s = []
    fold_weighted_f1s = []
    fold_precisions = []
    fold_recalls = []
    fold_aucs = []
    aggregated_cm = np.zeros((3, 3), dtype=int)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Strict featurization ordering: fit scaler ONLY on training fold
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        clf = create_model_fn()
        clf.fit(X_train_scaled, y_train)

        preds = clf.predict(X_val_scaled)
        probs = clf.predict_proba(X_val_scaled)

        acc = accuracy_score(y_val, preds)
        mf1 = f1_score(y_val, preds, average="macro", zero_division=0)
        wf1 = f1_score(y_val, preds, average="weighted", zero_division=0)
        prec = precision_score(y_val, preds, average="macro", zero_division=0)
        rec = recall_score(y_val, preds, average="macro", zero_division=0)

        try:
            auc = roc_auc_score(y_val, probs, multi_class="ovr", average="macro")
        except Exception:
            auc = 0.0

        cm = confusion_matrix(y_val, preds, labels=[0, 1, 2])
        aggregated_cm += cm

        fold_accuracies.append(acc)
        fold_macro_f1s.append(mf1)
        fold_weighted_f1s.append(wf1)
        fold_precisions.append(prec)
        fold_recalls.append(rec)
        fold_aucs.append(auc)

    return {
        "model_name": model_name,
        "accuracy_mean": float(np.mean(fold_accuracies)),
        "accuracy_std": float(np.std(fold_accuracies)),
        "macro_f1_mean": float(np.mean(fold_macro_f1s)),
        "macro_f1_std": float(np.std(fold_macro_f1s)),
        "weighted_f1_mean": float(np.mean(fold_weighted_f1s)),
        "precision_mean": float(np.mean(fold_precisions)),
        "recall_mean": float(np.mean(fold_recalls)),
        "roc_auc_mean": float(np.mean(fold_aucs)),
        "confusion_matrix": aggregated_cm.tolist(),
        "fold_macro_f1s": [round(f, 4) for f in fold_macro_f1s],
    }


def train_production_pipeline(
    data_dir: str,
    output_model_path: str,
    output_onnx_path: str,
    output_scaler_path: str,
    output_report_path: str,
    cache_path: str,
    modality: str = "bvp",
    window_sec: float = 45.0,
    step_sec: float = 15.0,
    n_splits: int = 5,
):
    """
    End-to-End ML Training, Multi-Model Cross-Validation, and Production Export.
    """
    print("\n" + "=" * 75)
    print("      ACTIVE RECOVERY SUITE - WESAD ML TRAINING PIPELINE      ")
    print("=" * 75)

    # 1. Ingest Data
    df = load_or_extract_dataset(
        data_dir=data_dir,
        cache_path=cache_path,
        modality=modality,
        window_sec=window_sec,
        step_sec=step_sec,
    )

    # Validate schema
    missing_feats = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing_feats:
        raise ValueError(f"Dataset missing required biometric features: {missing_feats}")

    # Inspect class counts
    class_counts = df["target"].value_counts().sort_index()
    print("\n[Data Distribution]")
    for cls_idx, count in class_counts.items():
        desc = CLASS_DESCRIPTIONS.get(cls_idx, f"Class {cls_idx}")
        pct = (count / len(df)) * 100.0
        print(f"  {desc}: {count} samples ({pct:.1f}%)")

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df["target"].values.astype(int)
    groups = df["subject"].values

    # Handle any potential NaNs or Infs
    if np.isnan(X).any() or np.isinf(X).any():
        print("[Preprocessing] Sanitizing NaNs and Infs in feature matrix...")
        col_medians = np.nanmedian(X, axis=0)
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i]) | np.isinf(X[:, i])
            X[mask, i] = col_medians[i]

    # 2. Multi-Model Tournament (GroupKFold)
    print(f"\n[Model Tournament] Running {n_splits}-Fold GroupKFold Cross-Validation (by Subject)...")

    # Model 1: Random Forest Baseline
    def create_rf():
        return RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    rf_metrics = evaluate_model_cv("RandomForest (Baseline)", create_rf, X, y, groups, n_splits=n_splits)
    print(f"  -> RandomForest:  Acc={rf_metrics['accuracy_mean']:.4f} +/- {rf_metrics['accuracy_std']:.4f} | "
          f"Macro F1={rf_metrics['macro_f1_mean']:.4f} +/- {rf_metrics['macro_f1_std']:.4f} | "
          f"ROC-AUC={rf_metrics['roc_auc_mean']:.4f}")

    # Model 2: XGBoost Classifier
    tournament_results = [rf_metrics]
    best_model_type = "RandomForest"

    if XGBOOST_AVAILABLE:
        def create_xgb():
            return XGBClassifier(
                n_estimators=180,
                max_depth=5,
                learning_rate=0.06,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
            )

        xgb_metrics = evaluate_model_cv("XGBoost (Champion Candidate)", create_xgb, X, y, groups, n_splits=n_splits)
        print(f"  -> XGBoost:       Acc={xgb_metrics['accuracy_mean']:.4f} +/- {xgb_metrics['accuracy_std']:.4f} | "
              f"Macro F1={xgb_metrics['macro_f1_mean']:.4f} +/- {xgb_metrics['macro_f1_std']:.4f} | "
              f"ROC-AUC={xgb_metrics['roc_auc_mean']:.4f}")
        tournament_results.append(xgb_metrics)

        if xgb_metrics["macro_f1_mean"] >= rf_metrics["macro_f1_mean"]:
            best_model_type = "XGBoost"

    # Select Champion Model
    print(f"\n[CHAMPION] Model Selected: {best_model_type}")

    # 3. Train Production Champion Model on Full Dataset with StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if best_model_type == "XGBoost" and XGBOOST_AVAILABLE:
        champion_clf = create_xgb()
    else:
        champion_clf = create_rf()

    champion_clf.fit(X_scaled, y)

    # 4. Feature Importance Analysis
    if hasattr(champion_clf, "feature_importances_"):
        importances = champion_clf.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        print("\n[Biometric Feature Importance Ranking]")
        feature_importance_dict = {}
        for rank, idx in enumerate(sorted_indices, 1):
            feat_name = FEATURE_NAMES[idx]
            imp_val = float(importances[idx])
            feature_importance_dict[feat_name] = round(imp_val, 4)
            bar = "#" * int(imp_val * 60)
            print(f"  {rank:2d}. {feat_name:<20}: {imp_val:.4f}  {bar}")

    # 5. Export Scaler Parameters to Lightweight JSON
    scaler_params = {
        "feature_names": FEATURE_NAMES,
        "mean": [float(m) for m in scaler.mean_],
        "scale": [float(s) for s in scaler.scale_],
        "class_mapping": {0: "GREEN", 1: "YELLOW", 2: "RED"},
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_scaler_path)), exist_ok=True)
    with open(output_scaler_path, "w") as f:
        json.dump(scaler_params, f, indent=2)
    print(f"\n[Export] Saved feature scaler parameters to {output_scaler_path}")

    # 6. Export Joblib Model Bundle
    joblib_bundle = {
        "model": champion_clf,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "class_mapping": {0: "GREEN", 1: "YELLOW", 2: "RED"},
        "winner_name": best_model_type,
        "metrics": tournament_results,
        "training_samples": len(df),
        "subjects": list(df["subject"].unique()),
        "modality": modality,
        "window_sec": window_sec,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_model_path)), exist_ok=True)
    joblib.dump(joblib_bundle, output_model_path)
    print(f"[Export] Saved trained champion bundle to {output_model_path}")

    # 7. Export to ONNX via export_onnx module
    try:
        from export_onnx import convert_model_to_onnx, verify_onnx_parity
        onnx_model = convert_model_to_onnx(
            model=champion_clf,
            model_type=best_model_type,
            n_features=len(FEATURE_NAMES),
            output_path=output_onnx_path,
        )
        # Verify inference parity
        verify_onnx_parity(
            model=champion_clf,
            onnx_path=output_onnx_path,
            sample_features=X_scaled[:10],
        )
        print(f"[Export] Exported validated ONNX model to {output_onnx_path}")
    except Exception as e:
        print(f"[Export] ONNX export failed with error: {e}")
        print("  Run 'python export_onnx.py' independently to troubleshoot.")

    # 8. Export Training Metrics Report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "WESAD (Wearable Stress and Affect Detection)",
        "modality": modality,
        "window_duration_seconds": window_sec,
        "stride_seconds": step_sec,
        "total_windows": len(df),
        "subjects_count": int(df["subject"].nunique()),
        "subjects_list": list(df["subject"].unique()),
        "class_distribution": {int(k): int(v) for k, v in class_counts.items()},
        "champion_model": best_model_type,
        "tournament_evaluation": tournament_results,
        "feature_importance": feature_importance_dict,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_report_path)), exist_ok=True)
    with open(output_report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Export] Saved comprehensive metrics report to {output_report_path}")

    print("\n" + "=" * 75)
    print("      TRAINING AND ARTIFACT EXPORT COMPLETED SUCCESSFULLY!     ")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="WESAD Physiological Fatigue Classifier Training Pipeline")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=r"C:\Users\user\Downloads\WESAD\WESAD",
        help="Path to WESAD root directory containing S2, S3, ... folders",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="bvp",
        choices=["bvp", "ecg"],
        help="Signal modality to use: 'bvp' (Empatica E4 wrist PPG, 64Hz) or 'ecg' (RespiBAN chest, 700Hz)",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=45.0,
        help="Sliding window size in seconds (default: 45s matching mobile test)",
    )
    parser.add_argument(
        "--step-sec",
        type=float,
        default=15.0,
        help="Sliding window step in seconds (default: 15s)",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default="data/wesad_extracted_features.csv",
        help="Path to cache extracted feature CSV",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="models/fatigue_classifier.joblib",
        help="Path to save champion joblib model bundle",
    )
    parser.add_argument(
        "--output-onnx",
        type=str,
        default="models/fatigue_model.onnx",
        help="Path to save exported ONNX model",
    )
    parser.add_argument(
        "--output-scaler",
        type=str,
        default="models/scaler_params.json",
        help="Path to save JSON feature scaler parameters",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="models/training_metrics.json",
        help="Path to save JSON evaluation report",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of GroupKFold splits for cross-validation",
    )

    args = parser.parse_args()

    # Fallback to local data/WESAD if provided directory does not exist
    if not os.path.exists(args.data_dir):
        local_dir = os.path.join(os.path.dirname(__file__), "data", "WESAD")
        if os.path.exists(local_dir):
            args.data_dir = local_dir
        else:
            print(f"[Error] WESAD directory not found at: {args.data_dir}")
            print(f"Please specify --data-dir with the path to the WESAD folder.")
            sys.exit(1)

    train_production_pipeline(
        data_dir=args.data_dir,
        output_model_path=args.output_model,
        output_onnx_path=args.output_onnx,
        output_scaler_path=args.output_scaler,
        output_report_path=args.output_report,
        cache_path=args.cache_file,
        modality=args.modality,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        n_splits=args.cv_splits,
    )


if __name__ == "__main__":
    main()
