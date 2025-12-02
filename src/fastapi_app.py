import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import asyncio
import json

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from core.audio_preprocessing import run_pipeline_streaming

# ---- Setup ----
logger = logging.getLogger("medical_voice_assistant")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).resolve().parent
SAVE_DIR = BASE_DIR / "uploads" / "json"
AUDIO_DIR = BASE_DIR / "uploads" / "audio"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# ---- Models ----
class QuestionAnswer(BaseModel):
    question: str
    answer: Optional[str]
    needs_asking: bool
    category: str

class ProcessRequest(BaseModel):
    visit_id: str
    audio_path: Optional[str] = None
    language: str = "ar"
    patient_name: str = "no name"
    patient_id: str = "no id"
    save: bool = True
    is_conversation: bool = False

    @field_validator("audio_path")
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
    mode: str
    is_conversation: bool
    is_medical: bool
    classification: Optional[str]
    confidence: Optional[int]
    raw_text: Optional[str]
    refined_text: Optional[str]
    translated_text: Optional[str]
    questions: List[QuestionAnswer]
    reasoning: str
    meta: Dict[str, Any]


# ---- App ----
app = FastAPI(title="Medical Voice Assistant API", version="2.0.0")

# Serve static files (CSS, JS, images)
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    """Serve the frontend UI"""
    return FileResponse("frontend/index.html")

@app.get("/api/v1/health")
def health():
    return {"status": "ok"}

@app.post("/api/v1/process/upload/stream")
async def process_via_upload_stream(
    visit_id: str = Form(...),
    language: str = Form("ar"),
    patient_name: str = Form("no name"),
    patient_id: str = Form("no id"),
    save: bool = Form(True),
    is_conversation: bool = Form(False),
    features: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    """Upload an audio file and stream processing results in real-time."""
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=415, detail=f"Unsupported type: {file.content_type}")

    logger.info(f"Streaming processing for visit_id={visit_id}, is_conversation={is_conversation}")

    async def event_generator():
        try:
            async for event in run_pipeline_streaming(
                visit_id=visit_id,
                language=language,
                patient_name=patient_name,
                patient_id=patient_id,
                save=save,
                is_conversation=is_conversation,
                features=features,
                uploaded_file=file,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)  # Small delay for client processing
        except Exception as e:
            logger.exception(f"Error in streaming pipeline: {e}")
            error_event = {
                "phase": "error",
                "status": "error",
                "error": str(e)
            }
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=2222)