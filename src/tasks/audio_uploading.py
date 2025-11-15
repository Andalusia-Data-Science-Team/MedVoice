from celery_app import celery_app
from core.audio_preprocessing import run_pipeline
import asyncio
import logging
from urllib.parse import quote

logger = logging.getLogger(__name__)

@celery_app.task(
    bind=True,
    name="tasks.audio_uploading.upload_audio_files",
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 3, 'countdown': 60}
)
def upload_audio_files(
    self,
    visit_id: str,
    audio_path: str,
    language: str,
    patient_name: str,
    patient_id: str,
    features: str = None,
    save: bool = True
):
    """Celery task to process uploaded audio file asynchronously with progress updates."""
    try:
        result = asyncio.run(
            _upload_audio_files(
                self,
                visit_id,
                audio_path,
                language,
                patient_name,
                patient_id,
                features,
                save
            )
        )
        redis_key = f"celery-task-meta-{self.request.id}"
        encoded_key = quote(redis_key)
        redis_insight_url = f"http://localhost:5540/browser/data/default/{encoded_key}"

        result["redis_insight_link"] = redis_insight_url
        return result
    
    except Exception as e:
        logger.error(f"Task failed: {e}", exc_info=True)
        self.update_state(state="FAILURE", meta={"error": str(e)})
        return {"status": "FAILURE", "error": str(e)}


async def _upload_audio_files(
    task_instance,
    visit_id: str,
    audio_path: str,
    language: str,
    patient_name: str,
    patient_id: str,
    features: str = None,
    save: bool = True
):
    """
    Async Celery background task to process an uploaded audio file.
    Provides progress updates at key pipeline steps.
    """
    try:
        # --- STEP 1: Start ---
        task_instance.update_state(
            state="PROGRESS",
            meta={"step": "Starting", "progress": 5}
        )

        # --- STEP 2: Transcription ---
        task_instance.update_state(
            state="PROGRESS",
            meta={"step": "Transcribing audio", "progress": 25}
        )

        # Run your main async pipeline
        pipeline_result = await run_pipeline(
            visit_id=visit_id,
            language=language,
            patient_name=patient_name,
            patient_id=patient_id,
            features=features,
            save=save,
            audio_path=audio_path
        )

        # --- STEP 3: Validation ---
        task_instance.update_state(
            state="PROGRESS",
            meta={"step": "Validating transcription", "progress": 50}
        )

        # --- STEP 4: Finalizing ---
        task_instance.update_state(
            state="PROGRESS",
            meta={"step": "Finalizing output", "progress": 90}
        )

        # ✅ Just add status to the existing pipeline_result
        result = {
            "status": "SUCCESS",
            **pipeline_result  # This already has all the correct values!
        }

        # --- STEP 5: Success ---
        task_instance.update_state(state="SUCCESS", meta=result)
        logger.info(f"Audio processing completed successfully for visit {visit_id}")

        return result

    except Exception as e:
        logger.error(f"Audio upload task failed: {e}", exc_info=True)
        task_instance.update_state(state="FAILURE", meta={"error": str(e)})
        raise