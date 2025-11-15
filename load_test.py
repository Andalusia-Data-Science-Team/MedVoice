from locust import HttpUser, task, between
import os

# Path to your actual test audio file
AUDIO_FILE_PATH = "/home/ai/Workspace/Doha/Medical_voice_assistant/dr_amr_elkiki2.ogg"

if not os.path.exists(AUDIO_FILE_PATH):
    raise FileNotFoundError(f"{AUDIO_FILE_PATH} not found")

class MedicalVoiceUser(HttpUser):
    # Wait time between each user request
    wait_time = between(1, 3)
    @task
    def upload_audio(self):
        with open(AUDIO_FILE_PATH, "rb") as f:
            # Use correct filename and MIME type for .ogg
            files = {"file": ("dr_amr_elkiki2.ogg", f, "audio/ogg")}
            data = {
                "visit_id": "visit_test_001",
                "language": "en",  # set based on your input audio language
                "patient_name": "John Doe",
                "patient_id": "12345",
                "save": "true",  # avoid writing to disk for speed
            }

            response = self.client.post(
                "/api/v1/process/upload",
                files=files,
                data=data,
            )

            # Optional: log failed requests for debugging
            if response.status_code != 200:
                print(f"❌ Error {response.status_code}: {response.text[:200]}")