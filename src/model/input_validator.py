import json
import logging
import re
from model.llm_service import LLMService
from core.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalValidator:
    VALIDATION_PROMPT = """
You are a medical content classifier.
Determine if the following text contains medical content such as symptoms, diagnoses, treatments, medications, or other clinical information.

Text:
{text}

Respond ONLY in valid JSON (no explanations, no extra text).
JSON format example:
{{
  "classification": "MEDICAL",
  "confidence": 95
}}
"""

    @staticmethod
    def validate_medical_content(text: str) -> dict:
        try:
            formatted_prompt = MedicalValidator.VALIDATION_PROMPT.format(text=text)
            logger.info("Validating medical content with LLM")

            # Use the non-streaming method
            response = LLMService._call_llm_api(
                model_account="accounts/fireworks/models/deepseek-v3-0324",
                prompt=formatted_prompt,
                temperature=0.1
            )

            if not response:
                logger.warning("LLM returned empty response")
                return {"is_medical": False, "confidence": 0, "classification": None, "raw_response": None}

            response = response.strip()
            logger.info(f"LLM raw response:\n{response}")

            # Ensure we have a JSON-like string
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                json_text = match.group(0)
            else:
                # Attempt to wrap missing braces
                json_text = "{" + response.strip().strip(',') + "}"

            # Optional: fix unquoted keys
            json_text = re.sub(r'(\w+):', r'"\1":', json_text)  # converts classification: -> "classification":
            result_json = json.loads(json_text)

            classification = result_json.get("classification", "").upper()
            confidence = int(result_json.get("confidence", 0))

            return {
                "is_medical": classification == "MEDICAL",
                "confidence": confidence,
                "classification": classification,
                "method": "llm_validation",
                "raw_response": response
            }

        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            return {
                "is_medical": False,
                "confidence": 0,
                "classification": None,
                "raw_response": response if 'response' in locals() else None
            }