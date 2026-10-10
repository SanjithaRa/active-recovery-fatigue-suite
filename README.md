# Active Recovery, Joint Health, and Fatigue Management Suite
## Phase 1: Contactless Photoplethysmography (rPPG) & Autonomic Fatigue Analysis

An end-to-end prototype integrating real-time smartphone optical pulse extraction (Flutter/Dart) with scientific digital signal processing and heuristic autonomic fatigue decision modeling (Python/FastAPI).

---

## Architecture Overview

```
fullesterker/
├── recovery_backend/                 # Python FastAPI Scientific Service
│   ├── requirements.txt              # FastAPI, NumPy, SciPy, Uvicorn
│   ├── main.py                       # API routes, CORS, payload validation
│   ├── services/
│   │   ├── signal_processor.py       # 3rd-order Butterworth bandpass, peak detection, rMSSD
│   │   └── rules_engine.py           # Autonomic triage table & volume prescription
│   └── tests/
│       └── test_signal_processor.py  # Unit tests with synthetic physiological waveforms
│
├── recovery_mobile/                  # Flutter Mobile Client (iOS & Android)
│   ├── pubspec.yaml                  # camera, http, modern Flutter deps
│   ├── android/app/src/main/
│   │   └── AndroidManifest.xml       # Camera, flash, cleartext network permissions
│   └── lib/
│       ├── main.dart                 # App entry point, camera discovery, dark theme
│       ├── models/
│       │   └── recovery_result.dart  # Strongly-typed data model & color mapping
│       ├── services/
│       │   └── fatigue_api_service.dart # HTTP client with IP configuration
│       ├── widgets/
│       │   └── pulse_waveform_painter.dart # CustomPainter real-time glowing PPG wave
│       └── screens/
│           └── pulse_scanner_screen.dart # 5-stage state machine, camera lock, 45s timer
│
└── README.md
```

---

## Physiological & Digital Signal Processing (DSP) Pipeline

1. **Optical Acquisition**:
   - The smartphone rear flashlight illuminates the fingertip capillary beds.
   - The camera frame stream (`startImageStream`) samples at ~30 FPS.
   - Microvascular blood volume expansion during cardiac systole absorbs green/luminance light, creating an alternating optical photoplethysmogram (PPG).

2. **Camera Sensor Stabilization**:
   - Auto-Exposure (AE), Auto-Focus (AF), and Auto-White-Balance (AWB) are locked on initiation to eliminate AGC gain fluctuations and illumination artifacts.

3. **SciPy Filtering**:
   - Zero-phase 3rd-order Butterworth bandpass filter ($0.75 \text{ Hz} - 3.0 \text{ Hz}$, corresponding to $45 - 180 \text{ BPM}$).
   - Linear detrending removes DC thermal sensor drift and contact pressure shifts.

4. **Autonomic Metrics**:
   - Systolic peak detection with adaptive prominence and minimum beat distance.
   - Inter-beat interval (RR interval) computation with physiological artifact rejection ($333 \text{ ms} - 1500 \text{ ms}$).
   - **Heart Rate**: $\text{BPM} = \frac{60000}{\text{mean}(RR)}$.
   - **HRV rMSSD**: $\text{rMSSD} = \sqrt{\text{mean}(\Delta RR^2)}$.

5. **Heuristic Triage Table**:
   - **Green ($\ge 45 \text{ ms}$)**: Optimal vagal tone. Full CNS capacity (100% volume).
   - **Yellow ($25 - 45 \text{ ms}$)**: Moderate parasympathetic suppression. Reduce compound volume by 15-20%.
   - **Red ($< 25 \text{ ms}$)**: Severe autonomic depletion. Deload / active recovery only.
