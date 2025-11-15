import os
import json
import logging
import aiofiles
import time
from pathlib import Path
from typing import Optional, Dict, Any, AsyncGenerator
from fastapi import UploadFile

from core.config import Config
from model.speech_service import SpeechService
from model.input_validator import MedicalValidator
from model.llm_service import LLMService
from model.extract_features import ExtractFeature

# ---- Setup ----
logger = logging.getLogger("medical_voice_assistant")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
SAVE_DIR = UPLOADS_DIR / "json"
AUDIO_DIR = UPLOADS_DIR / "audio"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


async def save_json(payload: Dict[str, Any]) -> Path:
    """Asynchronously save the output JSON."""
    out_path = SAVE_DIR / f"{payload.get('visit_id', 'no_id')}.json"
    async with aiofiles.open(out_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(payload, ensure_ascii=False, indent=2))
    return out_path

DEFAULT_FEATURES = """{
    "chief_complaint": "string",
    "icd10_codes": ["string"],
    "history_of_illness": "string",
    "current_medication": "string",
    "imaging_results": "string",
    "plan": "string",
    "assessment": "string",
    "follow_up": "string"
    }"""

DEFAULT_CONVERSATION_FEATURES = """{
    "chief_complaint": "string",
    "icd10_codes": ["string"],
    "history_of_illness": "string",
    "current_medication": "string",
    "imaging_results": "string",
    "plan": "string",
    "assessment": "string",
    "follow_up": "string",
    "conversation_summary": "string"
    }"""

# ---- Streaming Pipeline ----
async def run_pipeline_streaming(
    *,
    visit_id: str,
    language: str,
    patient_name: str,
    patient_id: str,
    save: bool,
    is_conversation: bool = False,
    features: Optional[str] = None,
    uploaded_file: Optional[UploadFile] = None,
    audio_path: Optional[str] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Streaming pipeline that yields progress updates for each phase with word-by-word streaming.
    """
    temp_path = None
    final_payload = {}

    try:
        # --- Determine audio source ---
        if uploaded_file:
            suffix = Path(uploaded_file.filename).suffix or ".wav"
            temp_path = AUDIO_DIR / f"{visit_id}{suffix}"
            async with aiofiles.open(temp_path, "wb") as out:
                while chunk := await uploaded_file.read(8192):
                    await out.write(chunk)
            audio_path = str(temp_path)
            logger.info(f"Uploaded file saved: {audio_path}")
        elif not audio_path:
            raise ValueError("Either 'audio_path' or 'file' must be provided.")

        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        mode = "conversation" if is_conversation else "doctor"
        api_key = Config.SPEECH_API_KEY or Config.FIREWORKS_API_KEY

        # Initialize final payload with metadata
        final_payload = {
            "visit_id": visit_id,
            "source_audio": str(audio_path),
            "language": language,
            "patient_name": patient_name,
            "patient_id": patient_id,
            "mode": mode,
            "is_conversation": is_conversation,
            "meta": {"timings": {}}
        }

        # --- Phase 1: Speech-to-text (STREAMING) ---
        yield {
            "phase": "transcription",
            "status": "processing",
            "message": "Converting speech to text...",
            "stream_start": True
        }

        t0 = time.perf_counter()
        raw_text = ""
        meta = {}
        
        async for chunk, chunk_meta in SpeechService.transcribe_audio_stream(
            str(audio_path),
            api_key=api_key,
            language=language,
            preprocess=True,
        ):
            if chunk_meta:
                # First yield contains metadata
                meta = chunk_meta
            else:
                # Subsequent yields contain text chunks
                raw_text += chunk
                yield {
                    "phase": "transcription",
                    "status": "streaming",
                    "chunk": chunk
                }
        
        timing = time.perf_counter() - t0
        final_payload["meta"]["timings"]["speech_to_text"] = timing
        final_payload["raw_text"] = raw_text
        final_payload["meta"].update(meta or {})

        yield {
            "phase": "transcription",
            "status": "complete",
            "result": raw_text,
            "timing": timing
        }

        # --- Phase 2: Medical validation ---
        yield {
            "phase": "validation",
            "status": "processing",
            "message": "Validating medical content..."
        }

        t0 = time.perf_counter()
        validation = MedicalValidator.validate_medical_content(raw_text) or {}
        timing = time.perf_counter() - t0
        final_payload["meta"]["timings"]["validation"] = timing
        final_payload["is_medical"] = bool(validation.get("is_medical"))
        final_payload["classification"] = validation.get("classification")
        final_payload["confidence"] = validation.get("confidence")

        yield {
            "phase": "validation",
            "status": "complete",
            "result": {
                "is_medical": final_payload["is_medical"],
                "classification": final_payload["classification"],
                "confidence": final_payload["confidence"]
            },
            "timing": timing
        }

        # --- Phase 3: Refinement (STREAMING) ---
        yield {
            "phase": "refinement",
            "status": "processing",
            "message": f"Refining text ({mode} mode)...",
            "stream_start": True
        }

        t0 = time.perf_counter()
        refined_text = ""
        
        if language.lower().startswith("ar"):
            stream_generator = LLMService.refine_ar_transcription_stream(
                raw_text, api_key, is_conversation=is_conversation
            )
        else:
            stream_generator = LLMService.refine_en_transcription_stream(
                raw_text, api_key, is_conversation=is_conversation
            )
        
        async for chunk in stream_generator:
            refined_text += chunk
            yield {
                "phase": "refinement",
                "status": "streaming",
                "chunk": chunk
            }
        
        timing = time.perf_counter() - t0
        final_payload["meta"]["timings"]["refine_text"] = timing
        final_payload["refined_text"] = refined_text

        yield {
            "phase": "refinement",
            "status": "complete",
            "result": refined_text,
            "timing": timing
        }

        # --- Phase 4: Translation (STREAMING) ---
        yield {
            "phase": "translation",
            "status": "processing",
            "message": "Translating to English...",
            "stream_start": True
        }

        t0 = time.perf_counter()
        translated_text = ""
        
        if language.lower().startswith("en"):
            translated_text = refined_text
            yield {
                "phase": "translation",
                "status": "complete",
                "result": translated_text,
                "timing": 0
            }
        else:
            async for chunk in LLMService.translate_to_eng_stream(
                refined_text, api_key, is_conversation=is_conversation
            ):
                translated_text += chunk
                yield {
                    "phase": "translation",
                    "status": "streaming",
                    "chunk": chunk
                }
            
            timing = time.perf_counter() - t0
            final_payload["meta"]["timings"]["translation"] = timing
            final_payload["translated_text"] = translated_text

            yield {
                "phase": "translation",
                "status": "complete",
                "result": translated_text,
                "timing": timing
            }

        if not translated_text:
            translated_text = refined_text
        final_payload["translated_text"] = translated_text


        # --- Phase 5: Feature Extraction (STREAMING) ---
        # In your audio_preprocessing.py, update the extraction phase:

        yield {
            "phase": "extraction",
            "status": "processing",
            "message": "Extracting medical features...",
            "stream_start": True
        }

        t0 = time.perf_counter()
        extraction_json = ""

        # Determine features schema
        if is_conversation and not features:
            schema_text = DEFAULT_CONVERSATION_FEATURES
        else:
            schema_text = features or DEFAULT_FEATURES

        async for chunk in ExtractFeature.extract_stream(
            translated_text, schema_text, is_conversation=is_conversation
        ):
            extraction_json += chunk
            yield {
                "phase": "extraction",
                "status": "streaming",
                "chunk": chunk
            }

        timing = time.perf_counter() - t0
        final_payload["meta"]["timings"]["feature_extraction"] = timing

        # Parse extraction JSON
        try:
            extraction_data = json.loads(extraction_json.strip())
            final_payload["json_data"] = extraction_data.get("json_data", {})
            final_payload["extraction_reasoning"] = extraction_data.get("reasoning", "")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse extraction JSON: {e}")
            final_payload["json_data"] = {}
            final_payload["extraction_reasoning"] = f"Error parsing: {str(e)}"

        yield {
            "phase": "extraction",
            "status": "complete",
            "result": {
                "json_data": final_payload["json_data"],
                "reasoning": final_payload["extraction_reasoning"]
            },
            "timing": timing
        }

        # --- Phase 6: Question Generation (STREAMING) ---
        yield {
            "phase": "questions",
            "status": "processing",
            "message": "Generating medical questions...",
            "stream_start": True
        }

        t0 = time.perf_counter()
        questions_json = ""
        
        async for chunk in LLMService.generate_questions_stream(
            translated_text, api_key, is_conversation=is_conversation
        ):
            questions_json += chunk
            yield {
                "phase": "questions",
                "status": "streaming",
                "chunk": chunk
            }
        
        timing = time.perf_counter() - t0
        final_payload["meta"]["timings"]["question_generation"] = timing
        
        # Parse questions JSON
        try:
            questions_data = json.loads(questions_json.strip())
            final_payload["questions"] = questions_data.get("questions", [])
            final_payload["reasoning"] = questions_data.get("reasoning", "")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse questions JSON: {e}")
            final_payload["questions"] = []
            final_payload["reasoning"] = f"Error parsing: {str(e)}"

        yield {
            "phase": "questions",
            "status": "complete",
            "result": {
                "questions": final_payload["questions"],
                "reasoning": final_payload["reasoning"]
            },
            "timing": timing
        }

        # --- Calculate total time ---
        total_time = sum(final_payload["meta"]["timings"].values())
        final_payload["meta"]["timings"]["total"] = total_time

        # --- Save if requested ---
        if save:
            await save_json(final_payload)
            logger.info(f"Output saved for visit_id={visit_id} (mode: {mode})")

        # --- Final complete event ---
        yield {
            "phase": "complete",
            "status": "complete",
            "result": final_payload
        }

    except Exception as e:
        logger.exception(f"Error in streaming pipeline for {visit_id}: {e}")
        yield {
            "phase": "error",
            "status": "error",
            "error": str(e)
        }
        raise

    finally:
        if temp_path and not save:
            try:
                os.remove(temp_path)
                logger.info(f"Temporary file deleted: {temp_path}")
            except OSError:
                pass