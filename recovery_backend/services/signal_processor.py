"""
Signal Processing Pipeline for Contactless/Fingertip Photoplethysmography (PPG).

Applies digital signal processing (DSP) using SciPy:
- Signal detrending and 3rd-order Butterworth bandpass filtering (0.75 Hz - 3.0 Hz / 45 - 180 BPM)
- Adaptive systolic peak detection via scipy.signal.find_peaks
- Inter-beat (RR) interval extraction with physiological outlier suppression
- Heart Rate (BPM) and Root Mean Square of Successive Differences (rMSSD) HRV calculation
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy import signal


class SignalProcessingError(Exception):
    """Raised when optical signal is too noisy, short, or invalid for physiological analysis."""
    pass


@dataclass
class ProcessedPPGResult:
    heart_rate_bpm: float
    rmssd_ms: float
    rr_intervals_ms: List[float]
    peak_count: int
    duration_seconds: float
    sampling_rate_hz: float
    signal_quality_score: float
    filtered_signal_preview: List[float]
    filtered_signal: Optional[np.ndarray] = None
    raw_signal: Optional[np.ndarray] = None


class PPGSignalProcessor:
    """
    Scientific DSP pipeline for extracting cardiovascular and autonomic
    metrics from smartphone optical frame streams.
    """

    def __init__(
        self,
        lowcut_hz: float = 0.75,   # 45 BPM
        highcut_hz: float = 3.0,   # 180 BPM
        filter_order: int = 3,
        min_duration_sec: float = 8.0,
        nominal_fs: float = 30.0,
    ):
        self.lowcut_hz = lowcut_hz
        self.highcut_hz = highcut_hz
        self.filter_order = filter_order
        self.min_duration_sec = min_duration_sec
        self.nominal_fs = nominal_fs

    def process(
        self,
        raw_signals: List[float],
        timestamps_ms: List[float],
    ) -> ProcessedPPGResult:
        """
        Execute full DSP pipeline on input raw optical stream.

        :param raw_signals: List of green/luminance channel mean values per frame.
        :param timestamps_ms: List of corresponding frame timestamps in milliseconds.
        :return: ProcessedPPGResult with HR, rMSSD, and signal quality metrics.
        :raises SignalProcessingError: When signal fails physiological validation.
        """
        if not raw_signals or not timestamps_ms:
            raise SignalProcessingError("Input signal and timestamp streams cannot be empty.")

        n_samples = len(raw_signals)
        if n_samples != len(timestamps_ms):
            raise SignalProcessingError(
                f"Mismatch: received {n_samples} signal samples and {len(timestamps_ms)} timestamps."
            )

        if n_samples < 90:  # Minimum ~3 seconds even for coarse checks, recommended >= 15-45s
            raise SignalProcessingError(
                f"Signal stream too short ({n_samples} frames). Minimum 90 frames (~3-5 seconds) required."
            )

        # Convert to numpy arrays
        sig = np.array(raw_signals, dtype=np.float64)
        ts = np.array(timestamps_ms, dtype=np.float64)

        # Standardize timestamps: if epoch ms or monotonic ms, sort if needed
        sort_idx = np.argsort(ts)
        ts = ts[sort_idx]
        sig = sig[sort_idx]

        # Calculate time delta and sampling rate
        total_duration_ms = ts[-1] - ts[0]
        total_duration_sec = total_duration_ms / 1000.0

        if total_duration_sec < self.min_duration_sec:
            raise SignalProcessingError(
                f"Capture duration ({total_duration_sec:.1f}s) is shorter than recommended minimum ({self.min_duration_sec:.1f}s). "
                "Maintain steady finger placement for the full measurement window."
            )

        # Calculate instantaneous sampling rates and verify stability
        dt_sec = np.diff(ts) / 1000.0
        # Filter out 0 or negative dt if duplicate frames occurred
        valid_dt = dt_sec[dt_sec > 0.005]  # > 5ms (max 200 fps)
        if len(valid_dt) < 10:
            fs = self.nominal_fs
        else:
            median_dt = np.median(valid_dt)
            fs = 1.0 / median_dt if median_dt > 0 else self.nominal_fs

        # Sanity check sampling rate
        if fs < 10.0:
            raise SignalProcessingError(
                f"Effective sampling rate ({fs:.1f} Hz) is too low for pulse analysis. Ensure camera frame stream >= 15 FPS."
            )

        # Verify signal variance (detect finger off / flatline / static sensor)
        sig_std = np.std(sig)
        if sig_std < 1e-4:
            raise SignalProcessingError(
                "Optical signal lacks pulsatile modulation (zero variance). "
                "Verify rear camera lens and flashlight are securely covered by fingertip."
            )

        # Baseline DC optical intensity check
        dc_mean = float(np.mean(sig))
        if dc_mean < 15.0:
            raise SignalProcessingError(
                "Optical signal too dark. Rear camera flashlight was not illuminating fingertip tissue."
            )

        # Remove linear baseline trend caused by gradual thermal drift or contact pressure changes
        detrended = signal.detrend(sig)

        # Apply 3rd-order Butterworth bandpass filter (0.75 Hz - 3.0 Hz / 45 - 180 BPM)
        filtered = self._apply_butterworth_bandpass(detrended, fs)

        # 1. Optical Perfusion Index (AC/DC pulsatile ratio) validation
        # In real microvascular PPG, AC pulsatile amplitude is typically 0.15% to 15% of DC level.
        ac_amplitude = float(np.percentile(filtered, 95) - np.percentile(filtered, 5))
        perfusion_index_pct = (ac_amplitude / (dc_mean + 1e-6)) * 100.0

        if perfusion_index_pct < 0.10:
            raise SignalProcessingError(
                f"Weak pulsatile modulation (Perfusion Index {perfusion_index_pct:.2f}% < 0.10%). "
                "No arterial pulse detected. Ensure your fingertip is held firmly against the camera lens and flashlight."
            )

        if perfusion_index_pct > 25.0:
            raise SignalProcessingError(
                f"Excessive motion artifact detected (Modulation {perfusion_index_pct:.1f}%). "
                "Please hold your finger steady on the camera lens without pressing too hard or sliding."
            )

        # 2. Periodicity & Autocorrelation Validation
        # Real cardiac pulse waves show strong rhythmicity in autocorrelation lag [0.31s - 1.50s].
        # Random noise, camera sensor flicker, or room light lacks high autocorrelation.
        corr = np.correlate(filtered, filtered, mode="full")
        corr = corr[len(corr) // 2 :]
        corr = corr / (corr[0] + 1e-12)

        min_lag = max(int(0.31 * fs), 3)  # max ~193 BPM
        max_lag = min(int(1.50 * fs), len(corr) - 1)  # min ~40 BPM
        lags = corr[min_lag:max_lag]
        max_autocorr = float(np.max(lags)) if len(lags) > 0 else 0.0

        if max_autocorr < 0.38:
            raise SignalProcessingError(
                f"Optical stream lacks synchronized cardiac rhythm (Autocorrelation: {max_autocorr:.2f} < 0.38). "
                "Fingertip was not properly placed or moved. Keep finger completely still over camera and flash."
            )

        # 3. Spectral Signal-to-Noise Ratio (SNR) in Cardiac Band
        nperseg = min(len(filtered), int(max(4.0 * fs, 64)))
        freqs, psd = signal.welch(filtered, fs=fs, nperseg=nperseg)
        cardiac_mask = (freqs >= 0.75) & (freqs <= 3.0)

        if not np.any(cardiac_mask) or np.sum(psd) <= 0:
            raise SignalProcessingError("Frequency domain transformation failed: empty spectrum.")

        cardiac_psd = psd[cardiac_mask]
        cardiac_freqs = freqs[cardiac_mask]
        dominant_freq = cardiac_freqs[np.argmax(cardiac_psd)]

        # Peak power within +/- 0.22 Hz of dominant cardiac harmonic
        peak_neighborhood = (freqs >= dominant_freq - 0.22) & (freqs <= dominant_freq + 0.22)
        peak_power = np.sum(psd[peak_neighborhood])
        total_power = np.sum(psd)
        cardiac_snr = float(peak_power / (total_power + 1e-12))

        if cardiac_snr < 0.22:
            raise SignalProcessingError(
                f"Signal-to-noise ratio in cardiac band too low ({cardiac_snr:.2f} < 0.22). "
                "Ambient room light or motion dominated the stream. Re-position finger firmly over lens."
            )

        # 4. Systolic Peak Detection
        peak_indices, signal_quality = self._detect_systolic_peaks(filtered, fs)

        if len(peak_indices) < 4:
            raise SignalProcessingError(
                f"Only {len(peak_indices)} systolic peaks detected. Could not establish reliable cardiac rhythm. "
                "Ensure steady fingertip placement without pressing too firmly."
            )

        # Inter-Beat (RR) intervals in milliseconds
        peak_times_ms = ts[peak_indices]
        raw_rr_intervals = np.diff(peak_times_ms)

        # Outlier filtering: retain physiological heart rate range (40 - 200 BPM -> 300 - 1500 ms)
        clean_rr = self._filter_physiologic_rr(raw_rr_intervals)

        if len(clean_rr) < 4:
            raise SignalProcessingError(
                "Excessive artifact noise: could not extract sufficient consecutive physiological RR intervals. "
                "Avoid moving your finger during acquisition."
            )

        # 5. Heart Rate & RR Coefficient of Variation Validation
        mean_rr_ms = float(np.mean(clean_rr))
        heart_rate_bpm = 60000.0 / mean_rr_ms

        if heart_rate_bpm < 42.0 or heart_rate_bpm > 195.0:
            raise SignalProcessingError(
                f"Extracted heart rate ({heart_rate_bpm:.1f} BPM) is outside normal resting human bounds (42 - 195 BPM). "
                "Please verify correct finger placement on camera lens and re-scan."
            )

        rr_std = float(np.std(clean_rr))
        rr_cv = rr_std / (mean_rr_ms + 1e-6)
        if rr_cv > 0.35:
            raise SignalProcessingError(
                f"Rhythm too erratic for valid analysis (RR CV {rr_cv:.2f} > 0.35). "
                "Please hold your finger steadily without lifting or rolling."
            )

        if signal_quality < 40.0:
            raise SignalProcessingError(
                f"Optical signal quality score ({signal_quality:.0f}%) is too low for reliable autonomic metrics. "
                "Keep your finger firmly over lens and flash."
            )

        # Autonomic metric: rMSSD = sqrt(mean(diff(RR)^2))
        rr_diffs = np.diff(clean_rr)
        rmssd_ms = float(np.sqrt(np.mean(rr_diffs ** 2)))

        # Subsample filtered waveform for mobile preview visualizer (e.g., ~100 points)
        step = max(1, len(filtered) // 100)
        preview_signal = filtered[::step].tolist()

        return ProcessedPPGResult(
            heart_rate_bpm=round(heart_rate_bpm, 1),
            rmssd_ms=round(rmssd_ms, 2),
            rr_intervals_ms=[round(float(v), 1) for v in clean_rr],
            peak_count=len(peak_indices),
            duration_seconds=round(total_duration_sec, 1),
            sampling_rate_hz=round(fs, 1),
            signal_quality_score=round(signal_quality, 1),
            filtered_signal_preview=preview_signal,
            filtered_signal=filtered,
            raw_signal=sig,
        )

    def _apply_butterworth_bandpass(self, data: np.ndarray, fs: float) -> np.ndarray:
        """Applies zero-phase 3rd-order Butterworth bandpass filter."""
        nyquist = 0.5 * fs
        low = self.lowcut_hz / nyquist
        high = min(self.highcut_hz / nyquist, 0.95)  # Cap high cut below Nyquist

        if low <= 0 or low >= high or high >= 1.0:
            raise SignalProcessingError(
                f"Invalid filter boundaries for sampling rate {fs:.1f} Hz. "
                f"Lowcut={self.lowcut_hz}Hz, Highcut={self.highcut_hz}Hz."
            )

        b, a = signal.butter(self.filter_order, [low, high], btype="bandpass")
        # filtfilt applies forward and reverse passes for zero phase distortion
        filtered = signal.filtfilt(b, a, data)
        return filtered

    def _detect_systolic_peaks(self, filtered: np.ndarray, fs: float) -> Tuple[np.ndarray, float]:
        """
        Systolic peak detection with adaptive distance and prominence thresholds.
        Returns peak indices and a computed signal quality score (0 - 100).
        """
        # Minimum peak distance corresponding to max human heart rate ~ 190 BPM (approx 0.31s)
        min_distance = max(int(0.31 * fs), 3)

        # Dynamic prominence based on standard deviation of filtered signal
        sig_std = np.std(filtered)
        prominence = max(0.20 * sig_std, 1e-4)

        peaks, properties = signal.find_peaks(
            filtered,
            distance=min_distance,
            prominence=prominence,
        )

        # Signal Quality Metric computation
        # Evaluates:
        # 1. Peak height consistency (coefficient of variation of peak amplitudes)
        # 2. RR interval consistency (variance in step durations)
        if len(peaks) >= 4:
            peak_amplitudes = filtered[peaks]
            amp_cv = np.std(peak_amplitudes) / (np.mean(peak_amplitudes) + 1e-6)
            amp_cv = np.clip(amp_cv, 0.0, 1.5)

            peak_diffs = np.diff(peaks) / fs
            rr_cv = np.std(peak_diffs) / (np.mean(peak_diffs) + 1e-6)
            rr_cv = np.clip(rr_cv, 0.0, 1.0)

            # High quality means low coefficient of variation of peak heights and stable rhythm
            quality = 100.0 - (amp_cv * 35.0 + rr_cv * 45.0)
            quality = float(np.clip(quality, 20.0, 98.0))
        else:
            quality = 25.0

        return peaks, quality

    def _filter_physiologic_rr(self, rr_intervals: np.ndarray) -> np.ndarray:
        """
        Filter out non-physiological artifacts (ectopic beats or missed detections).
        Retains RR intervals between 333 ms (180 BPM) and 1500 ms (40 BPM).
        Also filters out sudden step deviations > 35% from running median.
        """
        # Step 1: Absolute physiological bounds
        valid_mask = (rr_intervals >= 333.0) & (rr_intervals <= 1500.0)
        bounded = rr_intervals[valid_mask]

        if len(bounded) < 3:
            return bounded

        # Step 2: Relative median deviation filter
        med = np.median(bounded)
        dev_mask = np.abs(bounded - med) <= (0.35 * med)
        filtered = bounded[dev_mask]

        return filtered
