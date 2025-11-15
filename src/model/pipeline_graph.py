from __future__ import annotations

import logging
from typing import TypedDict, Optional, Any, Dict

from langgraph.graph import StateGraph, START, END

from core.config import Config
from model.speech_service import SpeechService
from model.input_validator import MedicalValidator
from model.refine_text import RefineText
from model.translation import Translate
from model.extract_features import ExtractFeature  # ensure your file is named extract_features.py

logger = logging.getLogger(__name__)

# Default features schema (used by ExtractFeature.extract when caller doesn't supply one)
# You can tailor this to your exact extraction keys.
DEFAULT_FEATURES = """
"chief_complaint": "",
"icd10_codes": [],
"history_of_illness": "",
"current_medication": "",
"imaging_results": "",
"plan": "",
"assessment": "",
"follow_up": ""
"""

class PipelineState(TypedDict, total=False):
    # Inputs
    file_path: str
    language: str              # "ar" or "en"
    api_key: str
    features: str              # schema text for extractor

    # Outputs per stage
    raw_text: str
    is_medical: bool
    validation: Dict[str, Any]
    refined_text: str
    translated_text: str
    json_data: Dict[str, Any]
    reasoning: str

    # Error
    error: str


# ---- Nodes -----------------------------------------------------------------

def transcribe_node(state: PipelineState) -> PipelineState:
    file_path = state["file_path"]
    api_key = state.get("api_key") or Config.FIREWORKS_API_KEY
    language = state.get("language", "ar")

    text = SpeechService.transcribe_audio(file_path, api_key=api_key, language=language, preprocess=True)
    return {**state, "raw_text": text}

def validate_node(state: PipelineState) -> PipelineState:
    raw = state.get("raw_text", "")
    # If you want a simple keyword gate, you can replace this with your own logic.
    result = MedicalValidator.validate_medical_content(raw)
    classification = result.get("classification", "NON_MEDICAL")
    is_medical = classification == "MEDICAL"
    out = {
        "is_medical": is_medical,
        "validation": {
            "classification": classification,
            "confidence": result.get("confidence", 0.0),
            "raw_response": result,
        }
    }
    return {**state, **out}

def refine_node(state: PipelineState) -> PipelineState:
    raw = state.get("raw_text", "")
    language = state.get("language", "ar")

    refined = RefineText.refining_transcription(
        raw_text=raw,
        language=language
    )
    return {**state, "refined_text": refined}

def translate_node(state: PipelineState) -> PipelineState:
    refined = state.get("refined_text", "")

    translated = Translate.translate(
        refined_text=refined,
    )
    return {**state, "translated_text": translated}

def extract_node(state: PipelineState) -> PipelineState:
    # If language is Arabic, we expect translate_node to have run, else we use refined_text
    language = state.get("language", "ar")
    end_text = state.get("translated_text") if language == "ar" else state.get("refined_text", "")

    features_schema = state.get("features") or DEFAULT_FEATURES

    json_data, reasoning = ExtractFeature.extract(
        end_text=end_text,
        features=features_schema,
    )
    return {**state, "json_data": json_data, "reasoning": reasoning}


# ---- Graph Builder ---------------------------------------------------------

def build_pipeline() -> StateGraph:
    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("transcribe", transcribe_node)
    graph.add_node("validate", validate_node)
    graph.add_node("refine", refine_node)
    graph.add_node("maybe_translate", translate_node)  # only if language == "ar"
    graph.add_node("extract", extract_node)

    # Edges
    graph.add_edge(START, "transcribe")
    graph.add_edge("transcribe", "validate")

    # Conditional: if NOT medical -> END, else continue
    def on_validate_cond(state: PipelineState) -> str:
        return "refine" if state.get("is_medical", False) else "__stop__"

    graph.add_conditional_edges("validate", on_validate_cond, {
        "refine": "refine",
        "__stop__": END,
    })

    # Conditional: if Arabic -> translate, else skip to extract
    def on_refine_cond(state: PipelineState) -> str:
        return "translate" if state.get("language", "ar") == "ar" else "extract"

    graph.add_conditional_edges("refine", on_refine_cond, {
        "translate": "maybe_translate",
        "extract": "extract",
    })

    graph.add_edge("maybe_translate", "extract")
    graph.add_edge("extract", END)

    return graph


# ---- Runner (helper for FastAPI) ------------------------------------------

def stream_pipeline(file_path: str, language: str, api_key: Optional[str] = None, features: Optional[str] = None):
    """
    Helper that builds the graph and yields (step_name, payload_dict) events,
    suitable for SSE streaming in FastAPI.
    """
    api_key = api_key or Config.FIREWORKS_API_KEY
    graph = build_pipeline().compile()

    # Initial state
    state: PipelineState = {
        "file_path": file_path,
        "language": language,
        "api_key": api_key,
    }
    if features:
        state["features"] = features

    # The stream yields events for each node execution
    for event in graph.stream(state, stream_mode="updates"):
        # event is a dict like {"node_name": {...updated_state...}}
        for node_name, payload in event.items():
            # Yield friendly step names + minimal payloads for the client
            if node_name == "transcribe":
                yield "transcription", {"text": payload.get("raw_text", "")}
            elif node_name == "validate":
                yield "validation", {
                    "is_medical": payload.get("is_medical"),
                    "classification": payload.get("validation", {}).get("classification"),
                    "confidence": payload.get("validation", {}).get("confidence"),
                }
            elif node_name == "refine":
                yield "refinement", {"text": payload.get("refined_text", "")}
            elif node_name == "maybe_translate":
                yield "translation", {"text": payload.get("translated_text", "")}
            elif node_name == "extract":
                yield "feature_extraction", {
                    "json_data": payload.get("json_data", {}),
                    "reasoning": payload.get("reasoning", ""),
                }