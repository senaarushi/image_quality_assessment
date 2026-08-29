"""
FastAPI backend for the AI-powered image quality & defect detection app.

Endpoints:
  POST /api/analyze       -- upload an image, run full inference pipeline, persist + return result
  GET  /api/results/{id}  -- retrieve a single past analysis (with Grad-CAM regenerated on demand)
  GET  /api/history       -- paginated list of past analyses
  GET  /api/health        -- health/status check

Run from backend/:  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import shutil
import uuid
from datetime import datetime

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.db.models import AnalysisResult, SessionLocal, get_db, init_db
from app.api.schemas import AnalysisResponse, HistoryResponse, HistoryItem, IssueSchema
from app.ml.inference import QualityInferencePipeline, CLASS_TO_IDX

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_FILE_SIZE_MB = 15

app = FastAPI(
    title="AI Image Quality & Defect Detection API",
    description="Upload an image and receive a structured quality assessment "
                "(blur, exposure, noise, corruption detection) using a hybrid "
                "classical-features + CNN model.",
    version="1.0.0",
)

# CORS: allow the frontend (any origin by default for local/dev; restrict via
# env var in production deployments)
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

# Loaded once at startup, reused across requests (loading models per-request
# would be far too slow).
pipeline: QualityInferencePipeline | None = None


@app.on_event("startup")
def on_startup():
    global pipeline
    init_db()
    try:
        pipeline = QualityInferencePipeline()
        print("Inference pipeline loaded successfully.")
    except FileNotFoundError as e:
        # Let the app boot anyway so /health still works and reports the
        # problem clearly, instead of crashing the whole process.
        print(f"WARNING: could not load inference pipeline: {e}")
        pipeline = None


@app.get("/api/health")
def health_check():
    return {
        "status": "ok" if pipeline is not None else "degraded",
        "models_loaded": pipeline is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }


def _validate_upload(file: UploadFile, contents: bytes):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f}MB). Max allowed is {MAX_FILE_SIZE_MB}MB.",
        )


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Inference models are not loaded. Check server logs / run training scripts.",
        )

    contents = await file.read()
    _validate_upload(file, contents)

    # Save the upload to disk under a unique name
    ext = os.path.splitext(file.filename)[1].lower()
    stored_filename = f"{uuid.uuid4().hex}{ext}"
    stored_path = os.path.join(UPLOADS_DIR, stored_filename)
    with open(stored_path, "wb") as f:
        f.write(contents)

    # Decode with OpenCV; handle unreadable/corrupt files gracefully
    image_bgr = cv2.imread(stored_path)
    if image_bgr is None:
        os.remove(stored_path)
        raise HTTPException(
            status_code=422,
            detail="Could not decode image. File may be corrupted or not a valid image.",
        )

    try:
        result = pipeline.analyze(image_bgr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    db: Session = SessionLocal()
    try:
        record = AnalysisResult(
            filename=file.filename,
            stored_image_path=f"/uploads/{stored_filename}",
            quality_score=result["quality_score"],
            quality_label=result["quality_label"],
            issues=result["issues"],
            image_stats=result["image_stats"],
            model_breakdown=result["model_breakdown"],
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        return AnalysisResponse(
            id=record.id,
            filename=record.filename,
            image_url=record.stored_image_path,
            quality_score=record.quality_score,
            quality_label=record.quality_label,
            issues=[IssueSchema(**i) for i in record.issues],
            image_stats=record.image_stats,
            model_breakdown=record.model_breakdown,
            created_at=record.created_at,
            gradcam_available=len(record.issues) > 0,
        )
    finally:
        db.close()


@app.get("/api/results/{result_id}", response_model=AnalysisResponse)
def get_result(result_id: int):
    db: Session = SessionLocal()
    try:
        record = db.query(AnalysisResult).filter(AnalysisResult.id == result_id).first()
        if record is None:
            raise HTTPException(status_code=404, detail=f"No analysis result found with id={result_id}")
        return AnalysisResponse(
            id=record.id,
            filename=record.filename,
            image_url=record.stored_image_path,
            quality_score=record.quality_score,
            quality_label=record.quality_label,
            issues=[IssueSchema(**i) for i in record.issues],
            image_stats=record.image_stats,
            model_breakdown=record.model_breakdown,
            created_at=record.created_at,
            gradcam_available=len(record.issues) > 0,
        )
    finally:
        db.close()


@app.get("/api/results/{result_id}/gradcam")
def get_gradcam(result_id: int):
    """Generates (or re-generates) the Grad-CAM overlay for a past result on demand."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Inference models are not loaded.")

    db: Session = SessionLocal()
    try:
        record = db.query(AnalysisResult).filter(AnalysisResult.id == result_id).first()
        if record is None:
            raise HTTPException(status_code=404, detail=f"No analysis result found with id={result_id}")
        if not record.issues:
            raise HTTPException(status_code=404, detail="No detected issue to explain for this result.")

        stored_path = os.path.join(BASE_DIR, record.stored_image_path.lstrip("/"))
        image_bgr = cv2.imread(stored_path)
        if image_bgr is None:
            raise HTTPException(status_code=500, detail="Stored image could not be re-read for Grad-CAM.")

        from app.ml.gradcam import generate_gradcam_overlay
        top_issue_type = record.issues[0]["type"]
        target_idx = CLASS_TO_IDX[top_issue_type]

        overlay_data_uri = generate_gradcam_overlay(
            pipeline.cnn_model, image_bgr, target_idx, pipeline.cnn_model.parameters().__next__().device,
        )
        return {"gradcam_image": overlay_data_uri, "explained_class": top_issue_type}
    finally:
        db.close()


@app.get("/api/history", response_model=HistoryResponse)
def get_history(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    db: Session = SessionLocal()
    try:
        total = db.query(AnalysisResult).count()
        records = (
            db.query(AnalysisResult)
            .order_by(desc(AnalysisResult.created_at))
            .offset(offset)
            .limit(limit)
            .all()
        )
        return HistoryResponse(
            total=total,
            results=[HistoryItem.model_validate(r) for r in records],
        )
    finally:
        db.close()