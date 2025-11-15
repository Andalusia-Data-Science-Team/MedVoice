import logging
import aiofiles
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from core.audio_preprocessing import run_pipeline
from utils.metrics import setup_metrics
from tasks.audio_uploading import upload_audio_files

# ---- Setup ----
logger = logging.getLogger("medical_voice_assistant")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
SAVE_DIR = UPLOADS_DIR / "json"
AUDIO_DIR = UPLOADS_DIR / "audio"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# ---- Models ----
class ProcessRequest(BaseModel):
    visit_id: str
    audio_path: Optional[str] = None
    language: str = "ar"
    patient_name: str = "no name"
    patient_id: str = "no id"
    features: Optional[str] = None
    save: bool = True

    @validator("audio_path")
    def validate_path(cls, path):
        if path and not Path(path).exists():
            raise ValueError(f"Audio file not found: {path}")
        return path


class ProcessResponse(BaseModel):
    visit_id: str
    source_audio: str
    language: str
    patient_name: str
    patient_id: str
    is_medical: bool
    classification: Optional[str]
    confidence: Optional[int]
    raw_text: Optional[str]
    refined_text: Optional[str]
    translated_text: Optional[str]
    json_data: Dict[str, Any]
    reasoning: str
    meta: Dict[str, Any]


# ---- App ----
app = FastAPI(title="Medical Voice Assistant API", version="2.0.0")

# ---- Setup Prometheus metrics ----
setup_metrics(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
def health():
    return {"status": "ok"}


# ---- Endpoints ----
@app.post("/api/v1/process", response_model=ProcessResponse)
async def process_via_path(req: ProcessRequest):
    """Process an existing audio file on disk."""
    return await run_pipeline(
        visit_id=req.visit_id,
        language=req.language,
        patient_name=req.patient_name,
        patient_id=req.patient_id,
        features=req.features,
        save=req.save,
        audio_path=req.audio_path,
    )


@app.post("/api/v1/process/upload")
async def process_via_upload(
    visit_id: str = Form(...),
    language: str = Form("ar"),
    patient_name: str = Form("no name"),
    patient_id: str = Form("no id"),
    save: bool = Form(True),
    features: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    """Upload an audio file and process it."""
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=415, detail=f"Unsupported type: {file.content_type}")
    
    audio_path = AUDIO_DIR / f"{visit_id}{Path(file.filename).suffix}"
    async with aiofiles.open(audio_path, "wb") as out:
        while chunk := await file.read(8192):
            await out.write(chunk)

    # Send Celery background task
    task = upload_audio_files.delay(
        visit_id=visit_id,
        audio_path=str(audio_path),
        language=language,
        patient_name=patient_name,
        patient_id=patient_id,
        features=features,
        save=save
    )
    
    return {
        "task_id": task.id,
        "status": "submitted",
        "message": f"Audio uploaded and task started for {visit_id}",
    }
    # return await run_pipeline(
    #     visit_id=visit_id,
    #     language=language,
    #     patient_name=patient_name,
    #     patient_id=patient_id,
    #     features=features,
    #     save=save,
    #     uploaded_file=file,
    # )