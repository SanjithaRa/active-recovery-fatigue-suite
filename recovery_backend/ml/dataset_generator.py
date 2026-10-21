"""
Physiological Dataset Generator & Ingestion Module.

Synthesizes clinically and athletically grounded distributions of
photoplethysmogram (PPG) and heart rate variability (HRV) metrics
across autonomic recovery tiers (GREEN, YELLOW, RED), or ingests
empirical athlete logs.
"""

import os
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from ml.feature_extractor import BiometricFeatureExtractor


class DatasetGenerator:
    """
    Generates realistic synthetic datasets with physiological covariance
    across autonomic recovery tiers for ML classifier training.
    """

    @classmethod
    def generate_synthetic_dataset(
        cls,
        n_samples_per_class: int = 500,
        random_state: int = 42,
        output_csv_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic biometric dataset with realistic cross-feature correlations.

        Classes:
        - 0: GREEN (Optimal Autonomic Recovery, Vagal Dominance)
        - 1: YELLOW (Moderate Parasympathetic Depletion, Sympathetic Elevation)
        - 2: RED (Severe Autonomic Fatigue, Exhaustion, High Systemic Strain)
        """
        np.random.seed(random_state)
        records = []

        # 1. GREEN Tier: High vagal tone, high HRV, low resting HR, low stress index
        for _ in range(n_samples_per_class):
            hr = np.random.normal(56.0, 5.0)
            rr_mean = 60000.0 / hr
            rmssd = np.random.uniform(46.0, 95.0)
            sdnn = rmssd * np.random.uniform(1.1, 1.45)
            pnn50 = np.clip(np.random.normal(32.0, 8.0), 18.0, 65.0)
            pnn20 = np.clip(pnn50 + np.random.uniform(15.0, 30.0), 40.0, 90.0)
            cv_rr = sdnn / rr_mean

            sd1 = np.sqrt(0.5 * (rmssd ** 2))
            sd2 = np.sqrt(max(2.0 * (sdnn ** 2) - 0.5 * (rmssd ** 2), 1e-4))
            sd1_sd2_ratio = sd1 / (sd2 + 1e-6)
            stress_index = np.clip(np.random.normal(45.0, 15.0), 12.0, 80.0)

            lf_pwr = np.random.uniform(0.2, 0.5)
            hf_pwr = np.random.uniform(0.4, 0.8)
            lf_hf_ratio = lf_pwr / (hf_pwr + 1e-6)
            total_pwr = lf_pwr + hf_pwr + np.random.uniform(0.1, 0.3)

            perfusion_index = np.random.uniform(1.2, 3.5)
            rise_time = np.random.normal(110.0, 12.0)
            quality = np.clip(np.random.normal(90.0, 5.0), 75.0, 98.0)

            hr_std = float(np.clip(np.random.normal(5.0, 1.2), 2.0, 10.0))
            row = {
                "hr_mean": hr,
                "hr_std": hr_std,
                "rr_mean": rr_mean,
                "sdnn": sdnn,
                "rmssd": rmssd,
                "pnn50": pnn50,
                "pnn20": pnn20,
                "cv_rr": cv_rr,
                "sd1": sd1,
                "sd2": sd2,
                "sd1_sd2_ratio": sd1_sd2_ratio,
                "stress_index": stress_index,
                "lf_power": lf_pwr,
                "hf_power": hf_pwr,
                "lf_hf_ratio": lf_hf_ratio,
                "total_power": total_pwr,
                "lf_power_est": lf_pwr,
                "hf_power_est": hf_pwr,
                "total_power_est": total_pwr,
                "perfusion_index": perfusion_index,
                "pulse_rise_time_ms": rise_time,
                "signal_quality": quality,
                "recovery_tier": "GREEN",
                "label": 0,
            }
            records.append(row)

        # 2. YELLOW Tier: Moderate vagal tone, slightly elevated HR, moderate stress
        for _ in range(n_samples_per_class):
            hr = np.random.normal(70.0, 6.0)
            rr_mean = 60000.0 / hr
            rmssd = np.random.uniform(25.0, 44.9)
            sdnn = rmssd * np.random.uniform(1.05, 1.35)
            pnn50 = np.clip(np.random.normal(12.0, 4.0), 5.0, 22.0)
            pnn20 = np.clip(pnn50 + np.random.uniform(15.0, 25.0), 20.0, 55.0)
            cv_rr = sdnn / rr_mean

            sd1 = np.sqrt(0.5 * (rmssd ** 2))
            sd2 = np.sqrt(max(2.0 * (sdnn ** 2) - 0.5 * (rmssd ** 2), 1e-4))
            sd1_sd2_ratio = sd1 / (sd2 + 1e-6)
            stress_index = np.clip(np.random.normal(130.0, 35.0), 75.0, 240.0)

            lf_pwr = np.random.uniform(0.4, 0.7)
            hf_pwr = np.random.uniform(0.2, 0.45)
            lf_hf_ratio = lf_pwr / (hf_pwr + 1e-6)
            total_pwr = lf_pwr + hf_pwr + np.random.uniform(0.1, 0.25)

            perfusion_index = np.random.uniform(0.8, 2.2)
            rise_time = np.random.normal(125.0, 15.0)
            quality = np.clip(np.random.normal(86.0, 6.0), 70.0, 96.0)

            hr_std = float(np.clip(np.random.normal(8.0, 1.8), 3.0, 15.0))
            row = {
                "hr_mean": hr,
                "hr_std": hr_std,
                "rr_mean": rr_mean,
                "sdnn": sdnn,
                "rmssd": rmssd,
                "pnn50": pnn50,
                "pnn20": pnn20,
                "cv_rr": cv_rr,
                "sd1": sd1,
                "sd2": sd2,
                "sd1_sd2_ratio": sd1_sd2_ratio,
                "stress_index": stress_index,
                "lf_power": lf_pwr,
                "hf_power": hf_pwr,
                "lf_hf_ratio": lf_hf_ratio,
                "total_power": total_pwr,
                "lf_power_est": lf_pwr,
                "hf_power_est": hf_pwr,
                "total_power_est": total_pwr,
                "perfusion_index": perfusion_index,
                "pulse_rise_time_ms": rise_time,
                "signal_quality": quality,
                "recovery_tier": "YELLOW",
                "label": 1,
            }
            records.append(row)

        # 3. RED Tier: Low HRV, suppressed parasympathetic tone, high stress index, elevated HR
        for _ in range(n_samples_per_class):
            hr = np.random.normal(84.0, 8.0)
            rr_mean = 60000.0 / hr
            rmssd = np.random.uniform(8.0, 24.9)
            sdnn = rmssd * np.random.uniform(0.95, 1.25)
            pnn50 = np.clip(np.random.normal(2.5, 1.5), 0.0, 6.0)
            pnn20 = np.clip(pnn50 + np.random.uniform(5.0, 15.0), 3.0, 25.0)
            cv_rr = sdnn / rr_mean

            sd1 = np.sqrt(0.5 * (rmssd ** 2))
            sd2 = np.sqrt(max(2.0 * (sdnn ** 2) - 0.5 * (rmssd ** 2), 1e-4))
            sd1_sd2_ratio = sd1 / (sd2 + 1e-6)
            stress_index = np.clip(np.random.normal(380.0, 110.0), 190.0, 950.0)

            lf_pwr = np.random.uniform(0.5, 0.9)
            hf_pwr = np.random.uniform(0.08, 0.22)
            lf_hf_ratio = lf_pwr / (hf_pwr + 1e-6)
            total_pwr = lf_pwr + hf_pwr + np.random.uniform(0.05, 0.2)

            perfusion_index = np.random.uniform(0.4, 1.5)
            rise_time = np.random.normal(145.0, 18.0)
            quality = np.clip(np.random.normal(82.0, 7.0), 65.0, 95.0)

            hr_std = float(np.clip(np.random.normal(12.0, 2.5), 5.0, 25.0))
            row = {
                "hr_mean": hr,
                "hr_std": hr_std,
                "rr_mean": rr_mean,
                "sdnn": sdnn,
                "rmssd": rmssd,
                "pnn50": pnn50,
                "pnn20": pnn20,
                "cv_rr": cv_rr,
                "sd1": sd1,
                "sd2": sd2,
                "sd1_sd2_ratio": sd1_sd2_ratio,
                "stress_index": stress_index,
                "lf_power": lf_pwr,
                "hf_power": hf_pwr,
                "lf_hf_ratio": lf_hf_ratio,
                "total_power": total_pwr,
                "lf_power_est": lf_pwr,
                "hf_power_est": hf_pwr,
                "total_power_est": total_pwr,
                "perfusion_index": perfusion_index,
                "pulse_rise_time_ms": rise_time,
                "signal_quality": quality,
                "recovery_tier": "RED",
                "label": 2,
            }
            records.append(row)

        df = pd.DataFrame(records)
        # Shuffle dataset
        df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

        if output_csv_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
            df.to_csv(output_csv_path, index=False)
            print(f"Dataset generated ({len(df)} samples) and saved to {output_csv_path}")

        return df


if __name__ == "__main__":
    DatasetGenerator.generate_synthetic_dataset(
        n_samples_per_class=600,
        output_csv_path="data/fatigue_dataset.csv",
    )
