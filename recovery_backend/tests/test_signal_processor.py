"""
Tests for Photoplethysmography (PPG) Signal Processor and Rules Engine.
"""

import math
import numpy as np
import pytest

from services.signal_processor import PPGSignalProcessor, SignalProcessingError
from services.rules_engine import FatigueRulesEngine


def generate_synthetic_ppg(
    duration_sec: float = 15.0,
    fs: float = 30.0,
    base_hr_bpm: float = 65.0,
    respiratory_modulation_hz: float = 0.25,
    noise_amplitude: float = 0.05,
    seed: int = 42,
):
    """
    Synthesize realistic optical fingertip PPG wave:
    - Fundamental cardiac frequency (HR / 60)
    - 2nd harmonic dicrotic notch
    - Low-frequency respiratory baseline wander
    - Additive Gaussian high-frequency camera noise
    """
    np.random.seed(seed)
    n_samples = int(duration_sec * fs)
    t = np.linspace(0, duration_sec, n_samples)
    timestamps_ms = [val * 1000.0 for val in t]

    f_cardiac = base_hr_bpm / 60.0

    # PPG pulse shape: asymmetric systolic upstroke + dicrotic reflection notch
    cardiac_wave = (
        0.8 * np.sin(2 * np.pi * f_cardiac * t)
        + 0.35 * np.sin(4 * np.pi * f_cardiac * t + 0.5)
        + 0.15 * np.sin(6 * np.pi * f_cardiac * t + 1.2)
    )

    # Respiratory sinus arrhythmia (RSA) / baseline wander
    respiratory_wander = 0.2 * np.sin(2 * np.pi * respiratory_modulation_hz * t)

    # Gaussian optical noise
    noise = noise_amplitude * np.random.randn(n_samples)

    # Simulated DC mean luminance value of camera sensor
    dc_offset = 120.0
    raw_signals = (dc_offset + cardiac_wave + respiratory_wander + noise).tolist()

    return timestamps_ms, raw_signals


def test_ppg_signal_processor_synthetic_extraction():
    processor = PPGSignalProcessor(min_duration_sec=8.0)
    timestamps_ms, raw_signals = generate_synthetic_ppg(
        duration_sec=15.0,
        fs=30.0,
        base_hr_bpm=70.0,
    )

    result = processor.process(raw_signals=raw_signals, timestamps_ms=timestamps_ms)

    # Verify Heart Rate extracted is within physiological tolerance of target 70 BPM
    assert 60.0 <= result.heart_rate_bpm <= 80.0
    assert result.peak_count >= 10
    assert result.sampling_rate_hz >= 25.0
    assert result.rmssd_ms > 0.0
    assert result.signal_quality_score >= 50.0


def test_ppg_signal_processor_too_short():
    processor = PPGSignalProcessor(min_duration_sec=8.0)
    timestamps_ms, raw_signals = generate_synthetic_ppg(duration_sec=3.0, fs=30.0)

    with pytest.raises(SignalProcessingError) as exc_info:
        processor.process(raw_signals=raw_signals, timestamps_ms=timestamps_ms)

    assert "shorter than recommended minimum" in str(exc_info.value)


def test_ppg_signal_processor_zero_variance():
    processor = PPGSignalProcessor(min_duration_sec=8.0)
    n = 300
    timestamps_ms = [i * 33.3 for i in range(n)]
    raw_signals = [128.0] * n  # completely flat line

    with pytest.raises(SignalProcessingError) as exc_info:
        processor.process(raw_signals=raw_signals, timestamps_ms=timestamps_ms)

    assert "zero variance" in str(exc_info.value)


def test_ppg_signal_processor_noise_rejection():
    """Verify that pure optical noise (finger off or ambient lighting) is strictly rejected."""
    processor = PPGSignalProcessor(min_duration_sec=8.0)
    np.random.seed(42)
    n = 450
    timestamps_ms = [i * 33.33 for i in range(n)]
    raw_noise = (120.0 + np.random.randn(n) * 2.0).tolist()

    with pytest.raises(SignalProcessingError):
        processor.process(raw_signals=raw_noise, timestamps_ms=timestamps_ms)


def test_rules_engine_tiers():
    # Green tier (rMSSD >= 45 ms)
    p_green = FatigueRulesEngine.evaluate(rmssd_ms=58.2, heart_rate_bpm=58.0)
    assert p_green.recovery_tier == "GREEN"
    assert "100%" in p_green.volume_adjustment
    assert p_green.tier_color_hex == "#00E676"

    # Yellow tier (25 ms <= rMSSD < 45 ms)
    p_yellow = FatigueRulesEngine.evaluate(rmssd_ms=32.4, heart_rate_bpm=69.0)
    assert p_yellow.recovery_tier == "YELLOW"
    assert "15-20%" in p_yellow.volume_adjustment
    assert p_yellow.tier_color_hex == "#FFD600"

    # Red tier (rMSSD < 25 ms)
    p_red = FatigueRulesEngine.evaluate(rmssd_ms=18.5, heart_rate_bpm=82.0)
    assert p_red.recovery_tier == "RED"
    assert "Deload" in p_red.volume_adjustment
    assert p_red.tier_color_hex == "#FF1744"
