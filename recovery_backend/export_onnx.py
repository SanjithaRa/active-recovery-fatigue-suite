"""
ONNX Model Exporter and Numerical Parity Validator.

Converts trained Scikit-Learn (RandomForest) and XGBoost models into
standardized Open Neural Network Exchange (ONNX) format for cross-platform,
ultra-low-latency, offline execution on mobile devices (Flutter via onnxruntime_flutter)
and edge inference servers.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Tuple, Optional

import joblib
import numpy as np
import onnx
import onnxruntime as ort

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType as SklFloatTensorType
    SKL2ONNX_AVAILABLE = True
except ImportError:
    SKL2ONNX_AVAILABLE = False

try:
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType as OnnxMlFloatTensorType
    ONNXMLTOOLS_AVAILABLE = True
except ImportError:
    ONNXMLTOOLS_AVAILABLE = False


def convert_model_to_onnx(
    model: Any,
    model_type: str,
    n_features: int = 18,
    output_path: str = "models/fatigue_model.onnx",
    target_opset: int = 12,
) -> onnx.ModelProto:
    """
    Convert a trained scikit-learn or XGBoost classifier to an ONNX model.

    :param model: Trained classifier instance (RandomForestClassifier or XGBClassifier).
    :param model_type: 'XGBoost' or 'RandomForest'.
    :param n_features: Dimensionality of feature vector (default: 18).
    :param output_path: Destination path for .onnx binary.
    :param target_opset: ONNX operator set version (default: 12).
    :return: Serialized ONNX ModelProto object.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    is_xgb = "XGB" in type(model).__name__ or model_type.lower() == "xgboost"

    if is_xgb:
        if not ONNXMLTOOLS_AVAILABLE:
            raise ImportError("onnxmltools is required for XGBoost ONNX conversion. Run: pip install onnxmltools")
        initial_types = [("float_input", OnnxMlFloatTensorType([None, n_features]))]
        onnx_model = convert_xgboost(model, initial_types=initial_types, target_opset=target_opset)
    else:
        if not SKL2ONNX_AVAILABLE:
            raise ImportError("skl2onnx is required for scikit-learn ONNX conversion. Run: pip install skl2onnx")
        initial_types = [("float_input", SklFloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(model, initial_types=initial_types, target_opset=target_opset)

    # Save to disk
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"[ONNX Export] Successfully converted {type(model).__name__} -> {output_path}")
    print(f"[ONNX Export] Model file size: {os.path.getsize(output_path) / 1024:.1f} KB")
    return onnx_model


def verify_onnx_parity(
    model: Any,
    onnx_path: str,
    sample_features: np.ndarray,
    atol: float = 1e-4,
) -> Dict[str, Any]:
    """
    Validate numerical parity between Python native inference and ONNX Runtime.

    :param model: Original trained Python model.
    :param onnx_path: Path to serialized .onnx file.
    :param sample_features: 2D numpy array of test samples.
    :param atol: Absolute error tolerance for class probabilities.
    :return: Parity evaluation summary.
    """
    sample_float32 = sample_features.astype(np.float32)

    # 1. Native Python Prediction
    py_preds = model.predict(sample_float32)
    py_probs = model.predict_proba(sample_float32)

    # 2. ONNX Runtime Execution
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    onnx_outputs = session.run(None, {input_name: sample_float32})
    onnx_preds = onnx_outputs[0]
    onnx_raw_probs = onnx_outputs[1]

    # Handle dictionary of probabilities vs matrix in ONNX outputs
    if isinstance(onnx_raw_probs, list) and len(onnx_raw_probs) > 0 and isinstance(onnx_raw_probs[0], dict):
        onnx_probs = np.array([[d[0], d[1], d[2]] for d in onnx_raw_probs], dtype=np.float32)
    elif isinstance(onnx_raw_probs, np.ndarray):
        onnx_probs = onnx_raw_probs
    else:
        onnx_probs = np.array(onnx_raw_probs, dtype=np.float32)

    # Compute numerical differences
    max_prob_diff = float(np.max(np.abs(py_probs - onnx_probs)))
    labels_match = bool(np.array_equal(py_preds, onnx_preds))
    probs_within_tol = bool(max_prob_diff <= atol)

    print(f"[ONNX Parity] Verifying {len(sample_features)} test samples:")
    print(f"  Class Predictions Match: {labels_match} (Exact Agreement: {np.mean(py_preds == onnx_preds)*100:.1f}%)")
    print(f"  Max Probability Delta:   {max_prob_diff:.6f} (Tolerance: {atol})")
    print(f"  Numerical Parity Status: {'PASSED [CONFIRMED]' if probs_within_tol else 'WARNING - Exceeds atol'}")

    return {
        "labels_match": labels_match,
        "max_prob_delta": max_prob_diff,
        "parity_passed": probs_within_tol,
        "input_tensor_name": input_name,
        "output_tensor_names": output_names,
    }


def export_scaler_parameters(
    scaler: Any,
    feature_names: list,
    output_path: str = "models/scaler_params.json",
) -> Dict[str, Any]:
    """
    Export StandardScaler mean and scale vectors to a lightweight JSON file
    for edge execution without requiring scikit-learn on mobile.
    """
    params = {
        "feature_names": feature_names,
        "mean": [float(m) for m in scaler.mean_],
        "scale": [float(s) for s in scaler.scale_],
        "class_mapping": {0: "GREEN", 1: "YELLOW", 2: "RED"},
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"[Scaler Export] Saved standardizer parameters to {output_path}")
    return params


def main():
    parser = argparse.ArgumentParser(description="Export Trained Model to ONNX format")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/fatigue_classifier.joblib",
        help="Path to trained champion model joblib bundle",
    )
    parser.add_argument(
        "--output-onnx",
        type=str,
        default="models/fatigue_model.onnx",
        help="Path to output .onnx model file",
    )
    parser.add_argument(
        "--output-scaler",
        type=str,
        default="models/scaler_params.json",
        help="Path to output scaler JSON file",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version (default: 12)",
    )

    args = parser.parse_args()

    model_path = args.model_path
    if not os.path.exists(model_path):
        backend_rel = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
        if os.path.exists(backend_rel):
            model_path = backend_rel

    if not os.path.exists(model_path):
        print(f"[Error] Model bundle not found at {args.model_path}.")
        print("Please train the model first by running: python train_fatigue_model.py")
        sys.exit(1)

    output_onnx = args.output_onnx
    output_scaler = args.output_scaler
    if not os.path.isabs(output_onnx) and not os.path.exists(os.path.dirname(output_onnx)):
        output_onnx = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_onnx)
    if not os.path.isabs(output_scaler) and not os.path.exists(os.path.dirname(output_scaler)):
        output_scaler = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_scaler)

    print(f"[Loading] Loading model bundle from {model_path}...")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle.get("feature_names", [])
    winner_name = bundle.get("winner_name", type(model).__name__)

    # Export scaler params
    export_scaler_parameters(scaler, feature_names, output_path=output_scaler)

    # Convert model to ONNX
    convert_model_to_onnx(
        model=model,
        model_type=winner_name,
        n_features=len(feature_names),
        output_path=output_onnx,
        target_opset=args.opset,
    )

    # Parity verification with synthetic test probe
    probe_raw = np.random.randn(10, len(feature_names)).astype(np.float32)
    probe_scaled = scaler.transform(probe_raw).astype(np.float32)
    verify_onnx_parity(model, output_onnx, probe_scaled)


if __name__ == "__main__":
    main()
