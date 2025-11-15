from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):

    FIREWORKS_API_KEY: str = None
    GROQ_API_KEY: str = None

    speech: str = None
    refine: str = None
    translation: str = None
    extraction: str = None
    questions:str = None
    UPLOAD_FOLDER: str

    # Celery Configuration
    CELERY_BROKER_URL: str = None
    CELERY_RESULT_BACKEND: str = None
    CELERY_TASK_SERIALIZER: str = "json"
    CELERY_TASK_TIME_LIMIT: int = 600
    CELERY_TASK_ACKS_LATE: bool = False
    CELERY_WORKER_CONCURRENCY: int = 2
    CELERY_FLOWER_PASSWORD: str = None

    class Config:
        env_file = ".env"

def get_settings():
    return Settings()