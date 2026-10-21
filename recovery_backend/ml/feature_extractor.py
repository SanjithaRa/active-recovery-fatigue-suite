"""
Autonomic and Cardiorespiratory Feature Extraction Pipeline.

Extracts high-dimensional time-domain, non-linear geometric (Poincare),
spectral approximations, and optical PPG morphometric features from
cardiovascular time-series and inter-beat (RR) intervals.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from scipy import signal


class BiometricFeatureExtractor:
    """
    Extracts clinical and athletic autonomic recovery features
    from photoplethysmogram (PPG) waveforms and RR interval series.
    """

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

    @classmethod
    def extract_features(
        cls,
        rr_intervals_ms: List[float],
        filtered_signal: Optional[np.ndarray] = None,
        raw_signal: Optional[np.ndarray] = None,
        fs: float = 30.0,
        signal_quality: float = 85.0,
    ) -> Dict[str, float]:
        """
        Extract full feature dictionary from RR intervals and optical waveforms.

        :param rr_intervals_ms: List of clean RR/peak-to-peak intervals in milliseconds.
        :param filtered_signal: Butterworth filtered PPG signal array.
        :param raw_signal: Raw optical frame luminance array.
        :param fs: Sampling frequency in Hz.
        :param signal_quality: DSP signal quality confidence score (0 - 100).
        :return: Dictionary of extracted numerical features.
        """
        rr = np.array(rr_intervals_ms, dtype=np.float64)

        if len(rr) < 3:
            # Safe defaults if stream is too short
            return {name: 0.0 for name in cls.FEATURE_NAMES}

        # 1. Time-Domain Metrics
        rr_mean = float(np.mean(rr))
        hr_bpm_series = 60000.0 / (rr + 1e-6)
        hr_mean = float(np.mean(hr_bpm_series))
        hr_std = float(np.std(hr_bpm_series)) if len(rr) > 1 else 0.0
        sdnn = float(np.std(rr, ddof=1)) if len(rr) > 1 else float(np.std(rr))

        diff_rr = np.diff(rr)
        rmssd = float(np.sqrt(np.mean(diff_rr ** 2)))

        pnn50 = float(np.sum(np.abs(diff_rr) > 50.0) / len(diff_rr) * 100.0) if len(diff_rr) > 0 else 0.0
        pnn20 = float(np.sum(np.abs(diff_rr) > 20.0) / len(diff_rr) * 100.0) if len(diff_rr) > 0 else 0.0
        cv_rr = float(sdnn / (rr_mean + 1e-6))

        # 2. Geometric & Poincare Analysis (Non-linear HRV)
        # SD1: short-term vagal variability along line of identity
        sd1 = float(np.sqrt(0.5 * (rmssd ** 2)))

        # SD2: continuous long-term autonomic variability
        var_rr = np.var(rr, ddof=1) if len(rr) > 1 else np.var(rr)
        sd2_sq = 2.0 * var_rr - 0.5 * (rmssd ** 2)
        sd2 = float(np.sqrt(max(sd2_sq, 1e-4)))
        sd1_sd2_ratio = float(sd1 / (sd2 + 1e-6))

        # Baevsky's Stress Index (SI): SI = AMo / (2 * Mo * MxDMn)
        # High SI reflects acute sympathetic tension and strain
        stress_index = cls._calculate_baevsky_stress_index(rr)

        # 3. Spectral Frequency-Domain Approximations (LF, HF, LF/HF)
        lf_pwr, hf_pwr, lf_hf_ratio, tot_pwr = cls._estimate_spectral_powers(rr)

        # 4. PPG Morphometrics & Perfusion Index
        perfusion_index = cls._calculate_perfusion_index(filtered_signal, raw_signal)
        pulse_rise_time_ms = cls._estimate_rise_time(filtered_signal, fs)

        return {
            "hr_mean": round(hr_mean, 2),
            "hr_std": round(hr_std, 2),
            "rr_mean": round(rr_mean, 2),
            "sdnn": round(sdnn, 2),
            "rmssd": round(rmssd, 2),
            "pnn50": round(pnn50, 2),
            "pnn20": round(pnn20, 2),
            "cv_rr": round(cv_rr, 4),
            "sd1": round(sd1, 2),
            "sd2": round(sd2, 2),
            "sd1_sd2_ratio": round(sd1_sd2_ratio, 4),
            "stress_index": round(stress_index, 2),
            "lf_power": round(lf_pwr, 3),
            "hf_power": round(hf_pwr, 3),
            "lf_hf_ratio": round(lf_hf_ratio, 3),
            "total_power": round(tot_pwr, 3),
            "perfusion_index": round(perfusion_index, 4),
            "pulse_rise_time_ms": round(pulse_rise_time_ms, 2),
        }

    @staticmethod
    def _calculate_baevsky_stress_index(rr: np.ndarray) -> float:
        """
        Calculate Baevsky's Stress Index:
        SI = (AMo / (2 * Mo * MxDMn)) * 10000
        AMo: Mode amplitude (percentage of intervals belonging to mode bin)
        Mo: Mode interval in seconds
        MxDMn: Max - Min range in seconds
        """
        if len(rr) < 4:
            return 50.0

        rr_sec = rr / 1000.0
        range_sec = float(np.max(rr_sec) - np.min(rr_sec))
        if range_sec < 0.05:
            range_sec = 0.05

        # 50ms histogram bins
        bin_width = 0.05
        bins = np.arange(np.min(rr_sec), np.max(rr_sec) + bin_width, bin_width)
        if len(bins) < 2:
            bins = 2
        counts, edges = np.histogram(rr_sec, bins=bins)

        max_bin_idx = int(np.argmax(counts))
        amo = (counts[max_bin_idx] / len(rr_sec)) * 100.0  # Percentage
        mo = (edges[max_bin_idx] + edges[max_bin_idx + 1]) / 2.0  # Bin center in seconds

        si = (amo / (2.0 * mo * range_sec + 1e-6))
        return float(np.clip(si, 10.0, 1500.0))

    @staticmethod
    def _estimate_spectral_powers(rr: np.ndarray) -> tuple[float, float, float, float]:
        """
        Estimate LF (0.04 - 0.15 Hz) and HF (0.15 - 0.40 Hz) power
        using Lomb-Scargle periodogram on unevenly spaced RR intervals.
        """
        if len(rr) < 6:
            return 0.5, 0.5, 1.0, 1.0

        # Create cumulative time axis in seconds
        time_sec = np.cumsum(rr) / 1000.0
        time_sec -= time_sec[0]

        # Frequencies of interest: 0.04 Hz to 0.40 Hz
        freqs = np.linspace(0.04, 0.40, 100)
        angular_freqs = 2.0 * np.pi * freqs

        # Center RR intervals
        rr_centered = rr - np.mean(rr)

        # Lomb-Scargle periodogram
        pgram = signal.lombscargle(time_sec, rr_centered, angular_freqs, normalize=True)

        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs <= 0.40)

        lf_power = float(np.sum(pgram[lf_mask]))
        hf_power = float(np.sum(pgram[hf_mask]))
        tot_power = float(np.sum(pgram))

        lf_hf_ratio = float(lf_power / (hf_power + 1e-6))
        return lf_power, hf_power, lf_hf_ratio, tot_power

    @staticmethod
    def _calculate_perfusion_index(
        filtered: Optional[np.ndarray],
        raw: Optional[np.ndarray],
    ) -> float:
        """Calculate Perfusion Index (PI) = (AC pulsatile component / DC baseline) * 100%."""
        if filtered is None or raw is None or len(filtered) == 0 or len(raw) == 0:
            return 1.25

        ac_amplitude = float(np.percentile(filtered, 95) - np.percentile(filtered, 5))
        dc_baseline = float(np.mean(raw))

        if dc_baseline <= 0:
            return 1.25

        pi = (ac_amplitude / dc_baseline) * 100.0
        return float(np.clip(pi, 0.1, 20.0))

    @staticmethod
    def _estimate_rise_time(filtered: Optional[np.ndarray], fs: float) -> float:
        """Estimate systolic pulse rise time from foot to systolic peak in milliseconds."""
        if filtered is None or len(filtered) < int(fs * 2):
            return 120.0  # Nominal physiological rise time in ms

        # Find min and max points in a representative beat
        peaks, _ = signal.find_peaks(filtered, distance=int(0.4 * fs))
        valleys, _ = signal.find_peaks(-filtered, distance=int(0.4 * fs))

        rise_times = []
        for p in peaks:
            # find preceding valley
            prior_valleys = valleys[valleys < p]
            if len(prior_valleys) > 0:
                v = prior_valleys[-1]
                dt_ms = ((p - v) / fs) * 1000.0
                if 50.0 <= dt_ms <= 300.0:
                    rise_times.append(dt_ms)

        if rise_times:
            return float(np.median(rise_times))
        return 120.0
