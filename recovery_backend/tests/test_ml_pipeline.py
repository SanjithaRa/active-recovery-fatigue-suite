"""
Unit Tests for Machine Learning Subsystem.

Tests:
- Biometric & cardiorespiratory feature extraction
- Dataset generator schema and distribution integrity
- Training pipeline execution & featurization ordering
- Inference engine probability output and readiness score
"""

import numpy as np
import pytest

from ml.feature_extractor import BiometricFeatureExtractor
from ml.dataset_generator import DatasetGenerator
from ml.trainer import FatigueModelTrainer
from ml.inference import FatigueMLInferenceEngine


def test_feature_extractor_outputs_all_features():
    # Synthetic clean RR intervals (around 70 BPM -> ~857 ms with small jitter)
    np.random.seed(42)
    rr_intervals = [850.0 + np.random.normal(0, 30.0) for _ in range(30)]

    features = BiometricFeatureExtractor.extract_features(
        rr_intervals_ms=rr_intervals,
        fs=30.0,
        signal_quality=92.0,
    )

    assert len(features) == len(BiometricFeatureExtractor.FEATURE_NAMES)
    for name in BiometricFeatureExtractor.FEATURE_NAMES:
        assert name in features
        val = features[name]
        assert not np.isnan(val)
        assert not np.isinf(val)

    assert 40.0 <= features["hr_mean"] <= 120.0
    assert features["rmssd"] > 0.0
    assert features["sd1"] > 0.0
    assert features["sd2"] > 0.0


def test_dataset_generator():
    df = DatasetGenerator.generate_synthetic_dataset(n_samples_per_class=20, random_state=42)

    assert len(df) == 60
    assert set(df["label"].unique()) == {0, 1, 2}
    for feat in BiometricFeatureExtractor.FEATURE_NAMES:
        assert feat in df.columns
        assert df[feat].isnull().sum() == 0


def test_inference_engine_prediction():
    engine = FatigueMLInferenceEngine()
    rr_intervals = [800.0, 810.0, 790.0, 820.0, 805.0, 795.0, 815.0, 800.0]

    result = engine.predict(rr_intervals_ms=rr_intervals)

    assert "recovery_tier" in result
    assert result["recovery_tier"] in ["GREEN", "YELLOW", "RED"]
    assert "readiness_score" in result
    assert 0.0 <= result["readiness_score"] <= 100.0

    probs = result["model_probabilities"]
    prob_sum = probs["green_prob"] + probs["yellow_prob"] + probs["red_prob"]
    assert pytest.approx(prob_sum, abs=0.05) == 1.0


def test_trainer_end_to_end(tmp_path):
    models_dir = str(tmp_path / "models")
    trainer = FatigueModelTrainer(
        models_dir=models_dir,
        random_state=42,
    )

    # Use small subset for quick test execution
    report = trainer.train_and_evaluate()

    assert "champion_model" in report
    assert "test_metrics" in report
    assert report["test_metrics"]["test_macro_f1"] >= 0.70

    # Verify model artifact was created
    engine = FatigueMLInferenceEngine(model_path=f"{models_dir}/fatigue_classifier.joblib")
    assert engine.is_trained is True

    # Test inference on newly trained artifact
    pred = engine.predict(rr_intervals_ms=[900.0, 920.0, 890.0, 910.0, 930.0, 895.0])
    assert pred["recovery_tier"] in ["GREEN", "YELLOW", "RED"]
