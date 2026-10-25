"""
Standalone Inference Testing Script for Autonomic Nervous System Fatigue Classification.

Demonstrates how the production system (FastAPI backend or mobile ONNX Runtime)
consumes a raw 45-second optical photoplethysmogram (PPG) pulse array,
applies SciPy DSP filtering, extracts the 18 cardiorespiratory biometrics,
normalizes features, and executes inference using the exported ONNX model.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Tuple, Optional

import numpy as np
import onnxruntime as ort
import joblib

# Import signal extraction logic from train_fatigue_model
from train_fatigue_model import (
    extract_window_biometrics,
    FEATURE_NAMES,
    CLASS_NAMES,
    CLASS_DESCRIPTIONS,
)


def synthesize_sample_ppg_stream(
    duration_sec: float = 45.0,
    fs: float = 64.0,
    state: str = "green",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthesize realistic 45-second optical PPG pulse stream with
    calibrated autonomic modulation for testing.
    
    :param duration_sec: 45.0 seconds.
    :param fs: Sampling rate (64 Hz matching Empatica E4 wrist PPG).
    :param state: 'green' (rested baseline), 'yellow' (moderate strain), or 'red' (acute stress).
    :return: (timestamps_sec, raw_optical_signal).
    """
    np.random.seed(seed)
    n_samples = int(duration_sec * fs)
    t = np.linspace(0, duration_sec, n_samples)

    if state.lower() == "green":
        # Rested athlete: Lower resting HR (58 BPM), high rMSSD, prominent respiratory sinus arrhythmia
        base_hr_bpm = 58.0
        rsa_amp = 0.35
        noise_amp = 0.04
    elif state.lower() == "yellow":
        # Moderate strain: Intermediate HR (74 BPM), moderate rMSSD
        base_hr_bpm = 74.0
        rsa_amp = 0.20
        noise_amp = 0.06
    else:
        # Acute stress / exhaustion: Elevated HR (95 BPM), blunted rMSSD, diminished RSA
        base_hr_bpm = 95.0
        rsa_amp = 0.08
        noise_amp = 0.09

    f_cardiac = base_hr_bpm / 60.0

    # Optical pulse wave: systolic upstroke + dicrotic notch
    pulse = (
        0.80 * np.sin(2 * np.pi * f_cardiac * t)
        + 0.35 * np.sin(4 * np.pi * f_cardiac * t + 0.5)
        + 0.15 * np.sin(6 * np.pi * f_cardiac * t + 1.2)
    )

    # Autonomic respiratory modulation (0.25 Hz = 15 breaths/min)
    respiration = rsa_amp * np.sin(2 * np.pi * 0.25 * t)

    # Ambient optical baseline + high-frequency sensor noise
    dc_offset = 120.0
    noise = noise_amp * np.random.randn(n_samples)

    raw_signal = dc_offset + pulse + respiration + noise
    return t, raw_signal


def compute_neuromuscular_readiness(
    probs: np.ndarray,
    rmssd: float,
    stress_index: float,
) -> float:
    """
    Calculate continuous Neuromuscular Readiness Score (0 - 100%).
    Combines model class probabilities with raw rMSSD vagal tone.
    """
    p_green, p_yellow, p_red = float(probs[0]), float(probs[1]), float(probs[2])

    # Probability weighted readiness
    base_readiness = (p_green * 100.0) + (p_yellow * 58.0) + (p_red * 18.0)

    # Bonus / penalty based on rMSSD
    if rmssd >= 60.0:
        rmssd_adj = min(10.0, (rmssd - 60.0) * 0.2)
    elif rmssd < 25.0:
        rmssd_adj = -max(5.0, (25.0 - rmssd) * 0.4)
    else:
        rmssd_adj = 0.0

    score = float(np.clip(base_readiness + rmssd_adj, 0.0, 100.0))
    return round(score, 1)


def generate_prescriptions(tier: str, readiness_score: float) -> Dict[str, str]:
    """
    Synthesize volume, intensity, and orthopedic joint safety prescriptions.
    """
    if tier == "GREEN":
        return {
            "volume_adjustment": "100% Full Training Volume",
            "intensity_target": "High Neural Drive (85 - 100% 1RM) / Explosive Velocity",
            "joint_recommendation": "Standard Dynamic Warm-up. Green-lit for heavy compound spinal loading (Squats/Deadlifts).",
            "autonomic_state": "Parasympathetic Rested & Receptive",
        }
    elif tier == "YELLOW":
        return {
            "volume_adjustment": "75 - 80% Moderate Volume (Reduce working sets by 1-2)",
            "intensity_target": "Submaximal Hypertrophy / Aerobic Capacity (70 - 80% 1RM)",
            "joint_recommendation": "Extend soft-tissue preparation by 10 minutes. Moderate joint compressive forces.",
            "autonomic_state": "Mild Sympathetic Depletion / Transient Strain",
        }
    else:  # RED
        return {
            "volume_adjustment": "Deload / Active Recovery (50% Volume or Mobility-Only)",
            "intensity_target": "Low Central Nervous System Demand (< 60% 1RM) / Zone 2 Recovery",
            "joint_recommendation": "Eliminate axial spinal compressive loading. Focus on passive decompression & mobility.",
            "autonomic_state": "Acute Sympathetic Hyperarousal / High Central Fatigue",
        }


def run_pipeline_inference(
    raw_signal: np.ndarray,
    fs: float = 64.0,
    onnx_path: str = "models/fatigue_model.onnx",
    scaler_path: str = "models/scaler_params.json",
    joblib_path: Optional[str] = "models/fatigue_classifier.joblib",
) -> Dict[str, Any]:
    """
    Full end-to-end inference execution on a 45-second pulse stream.
    """
    print(f"\n[DSP Pipeline] Processing {len(raw_signal)} samples ({len(raw_signal)/fs:.1f}s @ {fs} Hz)...")

    # 1. DSP & Biometric Extraction
    feats = extract_window_biometrics(raw_signal, fs=fs)
    if feats is None:
        raise ValueError("DSP Extraction failed: signal was too noisy or contained insufficient physiological pulses.")

    print("[Biometric Extraction] Key Extracted Biomarkers:")
    print(f"  • Heart Rate:             {feats['hr_mean']} BPM (Std: {feats['hr_std']})")
    print(f"  • rMSSD (Vagal Tone):     {feats['rmssd']} ms")
    print(f"  • SDNN (Total Variance):  {feats['sdnn']} ms")
    print(f"  • pNN50:                  {feats['pnn50']}%")
    print(f"  • Baevsky Stress Index:   {feats['stress_index']}")
    print(f"  • LF/HF Ratio:            {feats['lf_hf_ratio']}")
    print(f"  • Perfusion Index:        {feats['perfusion_index']}%")

    # 2. Build feature vector
    feature_vector = np.array([[feats[name] for name in FEATURE_NAMES]], dtype=np.float32)

    # 3. Scale features using scaler_params.json
    with open(scaler_path, "r") as f:
        scaler_data = json.load(f)

    mean_vec = np.array(scaler_data["mean"], dtype=np.float32)
    scale_vec = np.array(scaler_data["scale"], dtype=np.float32)
    scaled_vector = (feature_vector - mean_vec) / scale_vec

    # 4. Run ONNX Inference
    print(f"[ONNX Runtime] Executing inference on {onnx_path}...")
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    onnx_out = session.run(None, {input_name: scaled_vector})
    pred_class_idx = int(onnx_out[0][0])
    raw_probs = onnx_out[1]

    if isinstance(raw_probs, list) and len(raw_probs) > 0 and isinstance(raw_probs[0], dict):
        probs = np.array([raw_probs[0][0], raw_probs[0][1], raw_probs[0][2]], dtype=np.float32)
    elif isinstance(raw_probs, np.ndarray):
        probs = raw_probs[0]
    else:
        probs = np.array(raw_probs[0], dtype=np.float32)

    tier = CLASS_NAMES[pred_class_idx]
    readiness = compute_neuromuscular_readiness(probs, feats["rmssd"], feats["stress_index"])
    prescriptions = generate_prescriptions(tier, readiness)

    # Cross-verify with joblib model if available
    if joblib_path and os.path.exists(joblib_path):
        bundle = joblib.load(joblib_path)
        joblib_probs = bundle["model"].predict_proba(bundle["scaler"].transform(feature_vector))[0]
        joblib_pred = int(np.argmax(joblib_probs))
        print(f"[Verification] Joblib prediction matches ONNX: {joblib_pred == pred_class_idx}")

    return {
        "status": "success",
        "predicted_tier": tier,
        "readiness_score": readiness,
        "confidence_percentage": round(float(np.max(probs)) * 100.0, 1),
        "class_probabilities": {
            "GREEN": round(float(probs[0]), 4),
            "YELLOW": round(float(probs[1]), 4),
            "RED": round(float(probs[2]), 4),
        },
        "biometrics": feats,
        "prescriptions": prescriptions,
    }


def main():
    parser = argparse.ArgumentParser(description="Test ONNX Fatigue Inference Pipeline")
    parser.add_argument(
        "--test-state",
        type=str,
        default="green",
        choices=["green", "yellow", "red"],
        help="Simulated physiological state to test: green, yellow, or red",
    )
    parser.add_argument(
        "--onnx-model",
        type=str,
        default="models/fatigue_model.onnx",
        help="Path to exported ONNX model",
    )
    parser.add_argument(
        "--scaler-params",
        type=str,
        default="models/scaler_params.json",
        help="Path to scaler_params.json",
    )
    parser.add_argument(
        "--wesad-sample",
        type=str,
        default=None,
        help="Optional path to a real WESAD subject pickle file (e.g. C:\\...\\S2.pkl) to test real biological slice",
    )

    args = parser.parse_args()

    # Fallback to local models dir if needed
    base_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_file = os.path.join(base_dir, args.onnx_model) if not os.path.exists(args.onnx_model) else args.onnx_model
    scaler_file = os.path.join(base_dir, args.scaler_params) if not os.path.exists(args.scaler_params) else args.scaler_params

    if not os.path.exists(onnx_file) or not os.path.exists(scaler_file):
        print(f"[Error] Required artifacts not found: {onnx_file} or {scaler_file}")
        print("Run 'python train_fatigue_model.py' to generate the model artifacts.")
        sys.exit(1)

    print("=" * 70)
    print("      ACTIVE RECOVERY SUITE - INFERENCE PIPELINE VERIFICATION      ")
    print("=" * 70)

    # Acquire pulse stream
    if args.wesad_sample and os.path.exists(args.wesad_sample):
        print(f"[Input] Extracting 45-second pulse slice from WESAD subject: {args.wesad_sample}")
        import pickle
        with open(args.wesad_sample, "rb") as f:
            wesad_data = pickle.load(f, encoding="latin1")
        bvp_sig = wesad_data["signal"]["wrist"]["BVP"].flatten()
        fs = 64.0
        # Slice 45 seconds from baseline
        raw_signal = bvp_sig[int(120 * fs) : int(165 * fs)]
    else:
        print(f"[Input] Synthesizing calibrated 45s optical PPG stream for condition: '{args.test_state.upper()}'")
        _, raw_signal = synthesize_sample_ppg_stream(duration_sec=45.0, fs=64.0, state=args.test_state)
        fs = 64.0

    # Execute inference
    result = run_pipeline_inference(
        raw_signal=raw_signal,
        fs=fs,
        onnx_path=onnx_file,
        scaler_path=scaler_file,
    )

    print("\n" + "=" * 70)
    print(f"  PREDICTED AUTONOMIC STATUS:   {result['predicted_tier']}")
    print(f"  NEUROMUSCULAR READINESS:      {result['readiness_score']}%")
    print(f"  CONFIDENCE:                   {result['confidence_percentage']}%")
    print("=" * 70)
    print(f"  Class Probabilities:  Green={result['class_probabilities']['GREEN']:.3f} | "
          f"Yellow={result['class_probabilities']['YELLOW']:.3f} | "
          f"Red={result['class_probabilities']['RED']:.3f}")
    print("\n[Prescriptions]")
    for k, v in result["prescriptions"].items():
        print(f"  • {k.replace('_', ' ').title()}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    main()
