import logging
from core.config import Config
from model.llm_service import LLMService

logger = logging.getLogger(__name__)

class ExtractFeature:
    """Extract medical features from transcribed content."""

    @staticmethod
    async def extract_stream(translated_text: str, schema_text: str, is_conversation: bool = False):
        """
        Stream feature extraction from translated text.
        
        Args:
            translated_text: The translated medical text
            schema_text: JSON schema defining features to extract
            is_conversation: Whether this is a doctor-patient conversation
            
        Yields:
            Text chunks as they're generated
        """
        try:
            import json
            
            # Parse features schema
            features_list = json.loads(schema_text) if isinstance(schema_text, str) else schema_text
            
            api_key = Config.FIREWORKS_API_KEY
            
            # Stream the extraction
            async for chunk in LLMService.extract_features_stream(
                translated_text=translated_text,
                features=features_list,
                api_key=api_key,
                is_conversation=is_conversation
            ):
                yield chunk
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            yield f'{{"error": "Feature extraction failed: {str(e)}"}}'

    @staticmethod
    def extract(translated_text: str, schema_text: str, is_conversation: bool = False):
        """
        Synchronous extraction (for backward compatibility).
        
        Returns:
            Tuple of (json_data, reasoning)
        """
        # This is a fallback for non-streaming contexts
        # In practice, you should use extract_stream in the pipeline
        logger.warning("Using synchronous extract - consider migrating to extract_stream")
        return {}, "Synchronous extraction not implemented - use streaming version"