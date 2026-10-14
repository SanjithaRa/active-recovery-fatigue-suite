"""
FastAPI Microservice: Autonomic Fatigue & Joint Recovery Intelligence Service.

Replaces static heuristic mock backend with a production Machine Learning
inference and training subsystem:
- High-dimensional biometric & cardiorespiratory feature extraction (Time-domain, Poincare, Spectral, Morphometrics)
- Multi-model evaluation (Logistic Regression, Random Forest, HistGradientBoosting, MLP)
- Continuous neuromuscular readiness scoring (0-100%) and class probability distributions
- Dynamic training volume prescription and joint preservation directives
- Autonomous training endpoint (/api/v1/train) & model status introspection (/api/v1/model-status)
"""

from datetime import datetime, timezone
import json
import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from services.signal_processor import PPGSignalProcessor, SignalProcessingError
from ml.inference import FatigueMLInferenceEngine
from ml.trainer import FatigueModelTrainer
from ml.feature_extractor import BiometricFeatureExtractor
from ml.data_parser import UniversalDataParser

app = FastAPI(
    title="Autonomic Fatigue & Recovery Intelligence Microservice",
    description="Scientific DSP optical pulse extraction, multi-domain HRV feature engineering & ML recovery triage.",
    version="2.0.0",
)

# Cross-Origin Resource Sharing (CORS) configured for independent frontend microservices
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate DSP Signal Processor and ML Inference Engine
processor = PPGSignalProcessor(
    lowcut_hz=0.75,     # 45 BPM
    highcut_hz=3.0,     # 180 BPM
    filter_order=3,
    min_duration_sec=6.0,
    nominal_fs=30.0,
)
inference_engine = FatigueMLInferenceEngine()


# --- Pydantic Data Contracts ---

class PulseDataRequest(BaseModel):
    timestamps_ms: List[float] = Field(
        ...,
        description="Chronological array of frame timestamps in milliseconds.",
        example=[0.0, 33.3, 66.7, 100.0],
    )
    raw_signals: List[float] = Field(
        ...,
        description="Mean luminance or green channel pixel intensities extracted per camera frame.",
        example=[124.5, 124.8, 125.1, 124.9],
    )
    device_info: Optional[str] = Field(
        default="Mobile Rear Camera PPG",
        description="Client device metadata or camera configuration identifier.",
    )
    subject_id: Optional[str] = Field(
        default="athlete_guest",
        description="Optional athlete or user identifier.",
    )


class ModelProbabilities(BaseModel):
    green_prob: float = Field(..., description="Probability of Optimal Autonomic Recovery")
    yellow_prob: float = Field(..., description="Probability of Moderate Parasympathetic Depletion")
    red_prob: float = Field(..., description="Probability of Severe Autonomic Fatigue")


class AnalysisResponse(BaseModel):
    status: str = "success"
    heart_rate_bpm: float
    rmssd_ms: float
    recovery_tier: str
    tier_color_hex: str
    state_label: str
    readiness_score: float = Field(..., description="Continuous Neuromuscular Readiness Score (0 - 100%)")
    volume_adjustment: str
    cns_readiness: str
    training_prescription: str
    recommended_activities: List[str]
    joint_health_advisory: str
    autonomic_summary: str
    model_probabilities: ModelProbabilities
    biometric_features: Dict[str, float]
    inference_engine: str
    signal_quality_score: float
    sampling_rate_hz: float
    duration_seconds: float
    peak_count: int
    rr_intervals_ms: List[float]
    timestamp: str


class TrainingRequest(BaseModel):
    data_path: Optional[str] = Field(
        default=None,
        description="Optional path to custom CSV dataset. If omitted, uses calibrated physiological distributions.",
    )
    random_state: int = Field(default=42, description="Random seed for reproducibility.")


class ErrorResponse(BaseModel):
    status: str = "error"
    error_code: str
    message: str
    recommendation: Optional[str] = None


@app.exception_handler(SignalProcessingError)
async def signal_processing_error_handler(request: Request, exc: SignalProcessingError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error_code="PHYSIOLOGICAL_DSP_ERROR",
            message=str(exc),
            recommendation=(
                "Ensure steady fingertip placement completely covering the rear camera lens and flashlight torch. "
                "Keep hand still for the full duration of the test."
            ),
        ).model_dump(),
    )


# --- Health & Metadata Endpoints ---

@app.get("/", tags=["Microservice Info"])
async def root():
    return {
        "service": "Autonomic Fatigue & Recovery Intelligence Microservice",
        "architecture": "Microservice (Decoupled Backend Unit)",
        "version": "2.0.0",
        "ml_engine": inference_engine.model_name,
        "is_model_trained": inference_engine.is_trained,
        "documentation": "/docs",
        "api_v1_base": "/api/v1",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "ok",
        "service": "recovery-backend-microservice",
        "ml_model_loaded": inference_engine.is_trained,
        "model_name": inference_engine.model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/model-status", tags=["Machine Learning"])
async def get_model_status():
    """Inspect the active champion ML model, validation metrics, and feature importances."""
    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models", "training_report.json"
    )
    report_data = None
    if os.path.exists(report_path):
        try:
            with open(report_path, "r") as f:
                report_data = json.load(f)
        except Exception:
            pass

    return {
        "is_trained": inference_engine.is_trained,
        "active_model": inference_engine.model_name,
        "model_path": inference_engine.model_path,
        "feature_count": len(inference_engine.feature_names),
        "features": inference_engine.feature_names,
        "training_report": report_data,
        "status": "ready" if inference_engine.is_trained else "awaiting_training",
    }


@app.get("/api/v1/features", tags=["Machine Learning"])
async def list_features():
    """Returns the comprehensive biometric and cardiorespiratory feature catalog."""
    return {
        "total_features": len(BiometricFeatureExtractor.FEATURE_NAMES),
        "feature_catalog": BiometricFeatureExtractor.FEATURE_NAMES,
    }


# --- Core Inference Endpoint (v1 & legacy alias) ---

@app.post(
    "/api/v1/analyze-fatigue",
    response_model=AnalysisResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Signal quality or acquisition failure"},
        400: {"model": ErrorResponse, "description": "Malformed request payload"},
    },
    tags=["Analysis"],
)
@app.post(
    "/api/analyze-fatigue",
    response_model=AnalysisResponse,
    include_in_schema=False,
)
async def analyze_fatigue(payload: PulseDataRequest):
    """
    Core ML Inference Pipeline:
    1. Digital Signal Processing: 3rd-order Butterworth bandpass filter & systolic peak detection
    2. Inter-beat (RR) intervals extraction and artifact cleaning
    3. Multi-domain feature engineering (Time-domain, Poincare geometry, Lomb-Scargle spectral, morphometrics)
    4. ML Classifier inference predicting recovery tier probabilities and continuous readiness score
    5. Synthesis of athletic training prescriptions and joint health advisories
    """
    if len(payload.raw_signals) != len(payload.timestamps_ms):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Array length mismatch: signals ({len(payload.raw_signals)}) vs timestamps ({len(payload.timestamps_ms)}).",
        )

    # 1. Execute SciPy DSP Pipeline
    dsp_result = processor.process(
        raw_signals=payload.raw_signals,
        timestamps_ms=payload.timestamps_ms,
    )

    # 2. Execute Production ML Inference
    ml_result = inference_engine.predict(
        rr_intervals_ms=dsp_result.rr_intervals_ms,
        filtered_signal=dsp_result.filtered_signal,
        raw_signal=dsp_result.raw_signal,
        fs=dsp_result.sampling_rate_hz,
        signal_quality=dsp_result.signal_quality_score,
    )

    # 3. Assemble and return structured microservice response contract
    return AnalysisResponse(
        status="success",
        heart_rate_bpm=dsp_result.heart_rate_bpm,
        rmssd_ms=dsp_result.rmssd_ms,
        recovery_tier=ml_result["recovery_tier"],
        tier_color_hex=ml_result["tier_color_hex"],
        state_label=ml_result["state_label"],
        readiness_score=ml_result["readiness_score"],
        volume_adjustment=ml_result["volume_adjustment"],
        cns_readiness=ml_result["cns_readiness"],
        training_prescription=ml_result["training_prescription"],
        recommended_activities=ml_result["recommended_activities"],
        joint_health_advisory=ml_result["joint_health_advisory"],
        autonomic_summary=ml_result["autonomic_summary"],
        model_probabilities=ModelProbabilities(
            green_prob=ml_result["model_probabilities"]["green_prob"],
            yellow_prob=ml_result["model_probabilities"]["yellow_prob"],
            red_prob=ml_result["model_probabilities"]["red_prob"],
        ),
        biometric_features=ml_result["biometric_features"],
        inference_engine=ml_result["inference_engine"],
        signal_quality_score=dsp_result.signal_quality_score,
        sampling_rate_hz=dsp_result.sampling_rate_hz,
        duration_seconds=dsp_result.duration_seconds,
        peak_count=dsp_result.peak_count,
        rr_intervals_ms=dsp_result.rr_intervals_ms,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# --- Autonomous Model Training Endpoint ---

@app.post("/api/v1/train", tags=["Machine Learning"])
async def trigger_training(payload: Optional[TrainingRequest] = None):
    """
    Trigger the end-to-end ML training pipeline:
    - Splits data (70% train / 15% val / 15% test) before scaling
    - Evaluates 4 candidate algorithms with 5-fold cross-validation
    - Selects champion based on Validation Macro F1 & ROC-AUC
    - Evaluates final generalization on unseen Test set
    - Persists model artifact and reloads in-memory inference engine
    """
    req = payload or TrainingRequest()
    trainer = FatigueModelTrainer(
        data_path=req.data_path,
        random_state=req.random_state,
    )

    report = trainer.train_and_evaluate()
    # Hot-reload the newly trained champion model into the running inference engine
    inference_engine.reload()

    return {
        "status": "success",
        "message": f"Successfully trained and activated champion model: {report['champion_model']}",
        "champion_model": report["champion_model"],
        "test_macro_f1": report["test_metrics"]["test_macro_f1"],
        "test_accuracy": report["test_metrics"]["test_accuracy"],
        "test_roc_auc": report["test_metrics"]["test_roc_auc"],
        "feature_importances": report.get("feature_importances", {}),
        "report_summary": report,
    }


@app.post("/api/v1/upload-dataset", tags=["Machine Learning"])
async def upload_dataset(
    file: UploadFile = File(..., description="Upload dataset as .pdf, .csv, .xlsx, .json, or .parquet"),
    train_now: bool = Form(True, description="Whether to automatically train the model on the uploaded dataset immediately"),
):
    """
    Upload and parse external clinical/athletic datasets:
    - Supports PDF medical/research reports (extracts tables and biometric values via pypdf)
    - Supports CSV, Excel (.xlsx), Parquet, and JSON
    - Automatically maps columns to standard biometric features
    - If train_now=True, triggers ML training tournament and hot-reloads model
    """
    try:
        content = await file.read()
        df = UniversalDataParser.parse_file(content, file.filename or "uploaded_file.csv")

        # Save standardized CSV
        os.makedirs("data", exist_ok=True)
        saved_csv_path = os.path.join("data", f"imported_{file.filename}.csv")
        df.to_csv(saved_csv_path, index=False)

        training_report = None
        if train_now:
            trainer = FatigueModelTrainer(data_path=saved_csv_path)
            training_report = trainer.train_and_evaluate()
            inference_engine.reload()

        return {
            "status": "success",
            "filename": file.filename,
            "rows_parsed": len(df),
            "columns": list(df.columns),
            "saved_dataset_path": saved_csv_path,
            "trained": train_now,
            "training_report": training_report,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process uploaded file '{file.filename}': {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
