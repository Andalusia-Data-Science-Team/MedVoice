import fireworks.client
import logging
import json
from typing import Optional, Type, AsyncGenerator
from pydantic import BaseModel, ValidationError

from utils import prompt as prompt_utils

# ---------------- Logger ---------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Pydantic Models ---------------- #
class QuestionAnswer(BaseModel):
    question: str
    answer: Optional[str]
    needs_asking: bool
    category: str

class GeneratedQuestions(BaseModel):
    questions: list[QuestionAnswer]
    reasoning: str

# ---------------- Pydantic Models ---------------- #
class ExtractedFeatures(BaseModel):
    json_data: dict
    reasoning: str

# ---------------- LLM Service ---------------- #
class LLMService:
    """Service wrapper around Fireworks LLM API for refinement, translation, and question generation."""

    # --- Public APIs --- #
    @staticmethod
    async def refine_en_transcription_stream(raw_text: str, api_key: str, is_conversation: bool = False):
        """Stream refined English text word by word."""
        async for chunk in LLMService.process_text_stream(
            text=raw_text, api_key=api_key, model="deepseek",
            prompt_type="refine_english", is_conversation=is_conversation
        ):
            yield chunk

    @staticmethod
    async def refine_ar_transcription_stream(raw_text: str, api_key: str, is_conversation: bool = False):
        """Stream refined Arabic text word by word."""
        async for chunk in LLMService.process_text_stream(
            text=raw_text, api_key=api_key, model="deepseek",
            prompt_type="refine_arabic", is_conversation=is_conversation
        ):
            yield chunk

    @staticmethod
    async def translate_to_eng_stream(refined_text: str, api_key: str, is_conversation: bool = False):
        """Stream translation word by word."""
        async for chunk in LLMService.process_text_stream(
            text=refined_text, api_key=api_key, model="deepseek",
            prompt_type="translate", is_conversation=is_conversation
        ):
            yield chunk

    @staticmethod
    async def generate_questions_stream(translated_text: str, api_key: str, is_conversation: bool = False):
        """Stream question generation (returns full JSON at end)."""
        async for chunk in LLMService.process_text_stream(
            text=translated_text, api_key=api_key, model="llama",
            prompt_type="generate_questions",
            pydantic_model=GeneratedQuestions,
            is_conversation=is_conversation
        ):
            yield chunk

    @staticmethod
    async def extract_features_stream(translated_text: str, features: list, api_key: str, is_conversation: bool = False):
        """Stream feature extraction (returns full JSON at end)."""
        async for chunk in LLMService.process_text_stream(
            text=translated_text, api_key=api_key, model="llama",
            prompt_type="extract_dynamic",
            features=features,
            pydantic_model=ExtractedFeatures,
            is_conversation=is_conversation
        ):
            yield chunk


    # --- Core Logic --- #
    @staticmethod
    async def process_text_stream(
        text: str,
        api_key: str,
        model: str,
        prompt_type: str,
        features: Optional[list] = None,
        pydantic_model: Optional[Type[BaseModel]] = None,
        is_conversation: bool = False,
    ) -> AsyncGenerator[str, None]:
        """Generic method to stream text processing word by word."""
        fireworks.client.api_key = api_key
        if model == "deepseek":
            model_account = "accounts/fireworks/models/deepseek-v3-0324"
        else:
            model_account = "accounts/fireworks/models/llama4-maverick-instruct-basic"

        # Generate prompt
        prompt = LLMService._get_prompt(
            prompt_type, text, features, is_conversation
        )
        logger.debug(f"Generated prompt (is_conversation={is_conversation}): {prompt[:200]}...")

        # Stream from LLM API
        async for chunk in LLMService._call_llm_api_stream(
            model_account=model_account,
            prompt=prompt,
            pydantic_model=pydantic_model,
        ):
            yield chunk

    # --- Private Helpers --- #
    @staticmethod
    def _get_prompt(prompt_type: str, text: str, features: Optional[list], is_conversation: bool):
        """Select prompt dynamically from utils.prompt based on mode and conversation flag."""
        mapping = {
            # --- English Refinement ---
            ("refine_english", False): prompt_utils.get_refine_english_prompt_deepseek,
            ("refine_english", True): prompt_utils.get_refine_english_prompt_deepseek_conversation,

            # --- Arabic Refinement ---
            ("refine_arabic", False): prompt_utils.get_refine_arabic_prompt_deepseek,
            ("refine_arabic", True): prompt_utils.get_refine_arabic_prompt_deepseek_conversation,

            # --- Translation ---
            ("translate", False): prompt_utils.get_translation_prompt_deepseek,
            ("translate", True): prompt_utils.get_translation_prompt_deepseek_conversation,

            # --- Question Generation ---
            ("generate_questions", False): prompt_utils.get_question_generation_prompt_llama,
            ("generate_questions", True): prompt_utils.get_question_generation_prompt_llama_conversation,
            
            # --- Feature Extraction ---
            ("extract_dynamic", False): lambda t: prompt_utils.get_dynamic_extraction_prompt_llama(t, features),
            ("extract_dynamic", True): lambda t: prompt_utils.get_dynamic_extraction_prompt_llama_conversation(t, features),
            
        }

        key = (prompt_type, is_conversation)
        func = mapping.get(key)

        if func is None:
            raise ValueError(f"Unsupported prompt type={prompt_type} with is_conversation={is_conversation}")

        return func(text)

    @staticmethod
    async def _call_llm_api_stream(
        model_account: str, 
        prompt: str, 
        pydantic_model: Optional[Type[BaseModel]] = None,
        temperature: float = 0
    ) -> AsyncGenerator[str, None]:
        """Stream LLM API response word by word."""
        
        params = {
            "model": model_account,
            "prompt": prompt,
            "max_tokens": 2000,
            "temperature": temperature,
            "stream": True,  # Enable streaming
        }

        if pydantic_model:
            params["response_format"] = {"type": "json_object", "schema": pydantic_model.schema()}

        try:
            response = fireworks.client.Completion.create(**params)
            
            buffer = ""
            for chunk in response:
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    delta = chunk.choices[0].text
                    if delta:
                        buffer += delta
                        # Yield word by word
                        words = buffer.split(' ')
                        # Keep last incomplete word in buffer
                        for word in words[:-1]:
                            if word:
                                yield word + ' '
                        buffer = words[-1] if words else ''
            
            # Yield remaining buffer
            if buffer:
                yield buffer
                
        except Exception as e:
            logger.error(f"Streaming API error: {e}")
            yield f"[Error: {str(e)}]"

    @staticmethod
    def _call_llm_api(
        model_account: str, 
        prompt: str, 
        pydantic_model: Optional[Type[BaseModel]] = None,
        temperature: float = 0
    ) -> Optional[str]:
        """Non-streaming LLM API call for synchronous operations."""
        
        params = {
            "model": model_account,
            "prompt": prompt,
            "max_tokens": 2000,
            "temperature": temperature,
            "stream": False,  # No streaming
        }

        if pydantic_model:
            params["response_format"] = {"type": "json_object", "schema": pydantic_model.schema()}

        try:
            response = fireworks.client.Completion.create(**params)
            
            if not response.choices or not response.choices[0].text.strip():
                logger.warning("LLM returned empty response")
                return None

            raw_output = response.choices[0].text.strip()

            if pydantic_model:
                try:
                    parsed_output = json.loads(raw_output)
                    validated_output = pydantic_model(**parsed_output)
                    return json.dumps(validated_output.dict())
                except (json.JSONDecodeError, ValidationError) as e:
                    logger.error(f"Structured output validation failed: {e}")
                    return None

            return raw_output
                
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return None