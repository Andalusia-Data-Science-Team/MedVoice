FIREWORKS_API_KEY = ""
GROQ_API_KEY = ""

speech = ""
refine = ""
translation = ""
extraction = ""
questions = ""
UPLOAD_FOLDER = "uploads"

# ========================= Celery Task Queue Config =========================
CELERY_BROKER_URL="amqp://voice_assistant:voice_assistant_1811@rabbitmq:5672/voice_vhost"
CELERY_RESULT_BACKEND="redis://:voice_assistant_1811@redis:6379/0"
CELERY_TASK_SERIALIZER="json"
CELERY_TASK_TIME_LIMIT=600
CELERY_TASK_ACKS_LATE=False
CELERY_WORKER_CONCURRENCY=2
CELERY_FLOWER_PASSWORD="voice_flower_2222"