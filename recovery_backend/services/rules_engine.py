"""
Heuristic Autonomic Fatigue Rules Engine.

Translates physiological metrics (HRV rMSSD, Heart Rate, Signal Quality)
into actionable neuromuscular readiness tiers and training volume prescriptions.
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class FatiguePrescription:
    recovery_tier: str          # "GREEN" | "YELLOW" | "RED"
    tier_color_hex: str         # UI Accent Hex
    state_label: str            # Concise physiological state
    cns_readiness: str          # Neuromuscular status
    volume_adjustment: str      # Prescribed workload adjustment
    training_prescription: str  # Comprehensive coaching guidance
    recommended_activities: List[str]
    joint_health_advisory: str
    autonomic_summary: str


class FatigueRulesEngine:
    """
    Expert heuristic decision engine for athletic fatigue management,
    joint health, and central nervous system (CNS) capacity calibration.
    """

    GREEN_RMSSD_THRESHOLD = 45.0
    YELLOW_RMSSD_THRESHOLD = 25.0

    @classmethod
    def evaluate(
        cls,
        rmssd_ms: float,
        heart_rate_bpm: float,
        signal_quality_score: float = 85.0,
    ) -> FatiguePrescription:
        """
        Evaluate autonomic status based on rMSSD and secondary cardiac markers.

        :param rmssd_ms: Root Mean Square of Successive Differences in milliseconds.
        :param heart_rate_bpm: Resting or seated pulse rate in beats per minute.
        :param signal_quality_score: Confidence score from DSP processor.
        :return: FatiguePrescription containing tier and coaching directives.
        """
        if rmssd_ms >= cls.GREEN_RMSSD_THRESHOLD:
            return FatiguePrescription(
                recovery_tier="GREEN",
                tier_color_hex="#00E676",  # Vibrant Electric Emerald
                state_label="Optimal Autonomic Recovery",
                cns_readiness="Full Neuromuscular Drive (100%)",
                volume_adjustment="100% Volume (Standard / Overload)",
                training_prescription=(
                    "Parasympathetic dominance and high vagal tone detected. "
                    "Central Nervous System (CNS) is primed for heavy axial loading, maximal rate of force development (RFD), "
                    "and compound barbell movements. You are cleared for planned high-intensity training."
                ),
                recommended_activities=[
                    "Heavy primary compound lifts (Squats, Deadlifts, Overhead Press)",
                    "Maximal power output & plyometric drills",
                    "High-threshold motor unit recruitment sets (RPE 8.5 - 9.5)",
                    "Standard joint loading with progressive overload",
                ],
                joint_health_advisory=(
                    "Connective tissue remodeling is in a regenerative state. Maintain standard dynamic warm-up "
                    "and full synovial fluid mobilization before working sets."
                ),
                autonomic_summary=(
                    f"rMSSD at {rmssd_ms:.1f} ms reflects strong vagal reactivation and low baseline stress. "
                    f"Resting HR is stable at {heart_rate_bpm:.1f} BPM."
                ),
            )

        elif rmssd_ms >= cls.YELLOW_RMSSD_THRESHOLD:
            return FatiguePrescription(
                recovery_tier="YELLOW",
                tier_color_hex="#FFD600",  # Athletic Cyber Gold
                state_label="Moderate Parasympathetic Depletion",
                cns_readiness="Compromised Neuromuscular Recovery",
                volume_adjustment="Reduce Compound Volume by 15-20%",
                training_prescription=(
                    "Moderate autonomic nervous system fatigue detected with elevated sympathetic drive. "
                    "Neuromuscular coordination is sub-optimal for absolute 1RM attempts. "
                    "Cap compound working sets at RPE <= 8, eliminate back-off drop sets, and transition to controlled hypertrophy tempos."
                ),
                recommended_activities=[
                    "Submaximal technical execution (65% - 75% 1RM)",
                    "Dumbbell & machine compound variations (sparing spinal axial loading)",
                    "Zone 2 steady-state aerobic flush (20-25 min) to stimulate lymphatic clearance",
                    "End-range isometric joint stability holds",
                ],
                joint_health_advisory=(
                    "Elevated systemic inflammation potential. Avoid ballistic joint shock and extreme heavy eccentric loading. "
                    "Focus on joint decompressive stretches and rotator cuff / hip capsule prep."
                ),
                autonomic_summary=(
                    f"rMSSD at {rmssd_ms:.1f} ms indicates suppressed parasympathetic brake. "
                    f"Pulse of {heart_rate_bpm:.1f} BPM suggests mild sympathetic compensation."
                ),
            )

        else:
            return FatiguePrescription(
                recovery_tier="RED",
                tier_color_hex="#FF1744",  # High-Alert Vivid Crimson
                state_label="Severe Autonomic Fatigue",
                cns_readiness="Suppressed CNS / Acute Systemic Strain",
                volume_adjustment="Deload / Active Recovery Only (-50% to -100%)",
                training_prescription=(
                    "Severe autonomic depletion and sympathetic hyperactivity detected. "
                    "Central drive and tendon elastic stiffness are significantly compromised, drastically elevating injury risk. "
                    "Strictly postpone high-intensity compound sets and max-effort training today. Prescribe active restorative flow."
                ),
                recommended_activities=[
                    "Gentle dynamic mobility & thoracic spine decompression",
                    "Low-impact Zone 1 walk (20-30 min outdoors in sunlight)",
                    "Contrast hydrotherapy or infrared sauna session",
                    "Parasympathetic downregulation: 4-7-8 diaphragmatic box breathing (10 min)",
                    "High-quality sleep extension & hydration priority",
                ],
                joint_health_advisory=(
                    "High susceptibility to acute tendonitis and ligamentous strain. Withhold heavy shear and compression forces. "
                    "Perform non-weight-bearing joint circles and myofascial release."
                ),
                autonomic_summary=(
                    f"Critically low rMSSD ({rmssd_ms:.1f} ms) combined with heart rate of {heart_rate_bpm:.1f} BPM "
                    "indicates pronounced systemic fatigue and unrecovered stress load."
                ),
            )
