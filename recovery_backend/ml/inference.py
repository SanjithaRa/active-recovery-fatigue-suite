"""
Production ML Inference Engine for Autonomic Fatigue Analysis.

Loads the champion trained model bundle, scales input feature vectors,
computes class probability distributions across recovery tiers,
calculates continuous neuromuscular readiness percentages, and
synthesizes training prescriptions.
"""

import os
from typing import Dict, Any, List, Optional
import joblib
import numpy as np

from ml.feature_extractor import BiometricFeatureExtractor


class FatigueMLInferenceEngine:
    """
    Inference service for real-time athletic fatigue triage and
    central nervous system (CNS) capacity calibration.
    """

    DEFAULT_MODEL_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "fatigue_classifier.joblib",
    )

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.bundle: Optional[Dict[str, Any]] = None
        self.model = None
        self.scaler = None
        self.feature_names = BiometricFeatureExtractor.FEATURE_NAMES
        self.class_mapping = {0: "GREEN", 1: "YELLOW", 2: "RED"}
        self.model_name = "Heuristic-Fallback"
        self._load_model()

    def _load_model(self) -> bool:
        """Attempt to load trained model bundle from disk."""
        if os.path.exists(self.model_path):
            try:
                self.bundle = joblib.load(self.model_path)
                self.model = self.bundle["model"]
                self.scaler = self.bundle["scaler"]
                self.feature_names = self.bundle.get("feature_names", self.feature_names)
                self.class_mapping = self.bundle.get("class_mapping", self.class_mapping)
                self.model_name = self.bundle.get("winner_name", "Trained-ML-Model")
                print(f"[ML Engine] Loaded champion model: {self.model_name} from {self.model_path}")
                return True
            except Exception as e:
                print(f"[ML Engine] Warning: Failed to load model bundle ({e}). Falling back to physiological heuristics.")
                return False
        else:
            print(f"[ML Engine] Model file {self.model_path} not found. Operating in awaiting-training standby mode.")
            return False

    @property
    def is_trained(self) -> bool:
        return self.model is not None and self.scaler is not None

    def reload(self) -> bool:
        """Reload model after training completion without restarting server."""
        return self._load_model()

    def predict(
        self,
        rr_intervals_ms: List[float],
        filtered_signal: Optional[np.ndarray] = None,
        raw_signal: Optional[np.ndarray] = None,
        fs: float = 30.0,
        signal_quality: float = 85.0,
    ) -> Dict[str, Any]:
        """
        Run inference on extracted optical PPG and RR time-series.
        """
        # 1. Extract high-dimensional biometric features
        features_dict = BiometricFeatureExtractor.extract_features(
            rr_intervals_ms=rr_intervals_ms,
            filtered_signal=filtered_signal,
            raw_signal=raw_signal,
            fs=fs,
            signal_quality=signal_quality,
        )

        # 2. Check if trained model is active
        if self.is_trained:
            feature_vector = np.array([[features_dict.get(name, 0.0) for name in self.feature_names]])
            X_scaled = self.scaler.transform(feature_vector)

            # Class probabilities: [P(GREEN), P(YELLOW), P(RED)]
            probs = self.model.predict_proba(X_scaled)[0]
            pred_class_idx = int(np.argmax(probs))
            tier = self.class_mapping.get(pred_class_idx, "YELLOW")

            p_green = float(probs[0])
            p_yellow = float(probs[1]) if len(probs) > 1 else 0.0
            p_red = float(probs[2]) if len(probs) > 2 else 0.0
            confidence = float(np.max(probs)) * 100.0

            # Continuous Neuromuscular Readiness Score (0 - 100%)
            readiness_score = (p_green * 100.0) + (p_yellow * 65.0) + (p_red * 25.0)
            inference_engine_used = f"ML-{self.model_name}"
        else:
            # Standby heuristic fallback until training script is executed
            rmssd = features_dict.get("rmssd", 35.0)
            if rmssd >= 45.0:
                tier = "GREEN"
                p_green, p_yellow, p_red = 0.88, 0.10, 0.02
                readiness_score = 92.0
            elif rmssd >= 25.0:
                tier = "YELLOW"
                p_green, p_yellow, p_red = 0.15, 0.75, 0.10
                readiness_score = 68.0
            else:
                tier = "RED"
                p_green, p_yellow, p_red = 0.03, 0.17, 0.80
                readiness_score = 32.0

            confidence = 85.0
            inference_engine_used = "Physiological-Heuristic (Model Awaiting Training)"

        # 3. Formulate Comprehensive Training Adaptation Directives
        prescription = self._build_prescription(
            tier=tier,
            readiness_score=readiness_score,
            features=features_dict,
            probabilities={"green": p_green, "yellow": p_yellow, "red": p_red},
            engine=inference_engine_used,
        )

        return prescription

    def _build_prescription(
        self,
        tier: str,
        readiness_score: float,
        features: Dict[str, float],
        probabilities: Dict[str, float],
        engine: str,
    ) -> Dict[str, Any]:
        """Synthesize training adjustments and physiological explanations."""
        hr = features.get("hr_mean", 65.0)
        rmssd = features.get("rmssd", 40.0)
        si = features.get("stress_index", 50.0)
        sd1_sd2 = features.get("sd1_sd2_ratio", 0.5)

        if tier == "GREEN":
            color_hex = "#00E676"
            state_label = "Optimal Autonomic Recovery"
            cns_readiness = f"High CNS Capacity ({readiness_score:.1f}% Readiness)"
            volume_adjustment = "100% Volume (Progressive Overload Cleared)"
            training_prescription = (
                f"ML model classified high vagal tone and parasympathetic dominance ({probabilities['green']*100:.1f}% confidence). "
                f"rMSSD is robust at {rmssd:.1f} ms with low Baevsky stress index ({si:.1f}). "
                "Central nervous system is primed for high-threshold motor unit recruitment, heavy multi-joint loading, and maximal effort work."
            )
            activities = [
                "Heavy compound axial loading (Squat, Deadlift, Press)",
                "Maximal rate of force development (RFD) & plyometrics",
                "High-intensity intervals / threshold training (RPE 8.5 - 9.5)",
                "Full training volume as scheduled",
            ]
            joint_advisory = (
                "Synovial fluid circulation and connective tissue repair are in optimal regenerative states. "
                "Perform standard dynamic multi-planar warm-up before loading."
            )
        elif tier == "YELLOW":
            color_hex = "#FFD600"
            state_label = "Moderate Parasympathetic Depletion"
            cns_readiness = f"Moderate CNS Fatigue ({readiness_score:.1f}% Readiness)"
            volume_adjustment = "Reduce Compound Volume by 15-20%"
            training_prescription = (
                f"ML model detected moderate autonomic fatigue with elevated sympathetic drive ({probabilities['yellow']*100:.1f}% probability). "
                f"rMSSD at {rmssd:.1f} ms and stress index of {si:.1f} indicate incomplete neural recovery. "
                "Cap compound working sets at RPE <= 8, avoid failure on barbell movements, and emphasize technical execution."
            )
            activities = [
                "Submaximal technique work (65% - 75% 1RM)",
                "Dumbbell & machine compound variations (limiting spinal compression)",
                "Zone 2 steady aerobic conditioning (20-25 min)",
                "Rotator cuff and hip capsule isometric joint holds",
            ]
            joint_advisory = (
                "Elevated systemic inflammatory markers possible. Avoid maximal eccentrics and ballistic joint shear forces. "
                "Prioritize joint decompressive stretching."
            )
        else:  # RED
            color_hex = "#FF1744"
            state_label = "Severe Autonomic Fatigue"
            cns_readiness = f"Suppressed CNS Drive ({readiness_score:.1f}% Readiness)"
            volume_adjustment = "Deload / Active Recovery Only (-50% to -100%)"
            training_prescription = (
                f"ML classifier flagged acute autonomic fatigue and sympathetic hyperactivity ({probabilities['red']*100:.1f}% probability). "
                f"Suppressed rMSSD ({rmssd:.1f} ms) combined with elevated Baevsky stress index ({si:.1f}) indicates acute systemic strain. "
                "High risk of injury and central overreaching. Postpone heavy lifting and prioritize active regeneration."
            )
            activities = [
                "Low-impact Zone 1 walk in natural sunlight (20-30 min)",
                "Dynamic thoracic spine and hip mobility flow",
                "Downregulation breathwork (4-7-8 diaphragmatic pacing, 10 min)",
                "Contrast hydrotherapy or infrared sauna",
                "Hydration priority and early sleep window",
            ]
            joint_advisory = (
                "Tendon stiffness and joint shock absorption are compromised. Strictly avoid heavy shear loading and high-impact landings."
            )

        return {
            "recovery_tier": tier,
            "tier_color_hex": color_hex,
            "state_label": state_label,
            "cns_readiness": cns_readiness,
            "readiness_score": round(readiness_score, 1),
            "volume_adjustment": volume_adjustment,
            "training_prescription": training_prescription,
            "recommended_activities": activities,
            "joint_health_advisory": joint_advisory,
            "autonomic_summary": (
                f"rMSSD: {rmssd:.1f} ms | Resting HR: {hr:.1f} BPM | Stress Index: {si:.1f} | "
                f"Autonomic Balance SD1/SD2: {sd1_sd2:.2f}."
            ),
            "model_probabilities": {
                "green_prob": round(probabilities["green"], 3),
                "yellow_prob": round(probabilities["yellow"], 3),
                "red_prob": round(probabilities["red"], 3),
            },
            "biometric_features": features,
            "inference_engine": engine,
        }
