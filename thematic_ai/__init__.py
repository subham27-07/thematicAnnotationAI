"""LLM-assisted thematic annotation of moderation justifications.

The pipeline is backend-agnostic: the same codebook, prompt, schema and output
format are used whether the annotator is OpenAI GPT-5.1 or a local Qwen model
served by Ollama. Runs are distinguished by output filename, never by column
name.
"""

from .config import RunConfig
from .codebook import Codebook, load_codebook
from .data import load_units, load_human_annotations, load_adjudicated
from .backends import get_backend, LLMBackend
from .prompts import build_messages, build_system_prompt, PromptVariant
from .annotate import annotate_units, AnnotationRun
from .evaluate import (
    evaluate_run,
    human_baseline,
    agreement_with_each_coder,
    compare_to_human_reliability,
    confidence_sweep,
    ceiling_analysis,
)
from .pipeline import Workspace, load_workspace, make_system_prompt

__all__ = [
    "RunConfig",
    "Codebook",
    "load_codebook",
    "load_units",
    "load_human_annotations",
    "load_adjudicated",
    "get_backend",
    "LLMBackend",
    "build_messages",
    "build_system_prompt",
    "PromptVariant",
    "annotate_units",
    "AnnotationRun",
    "evaluate_run",
    "human_baseline",
    "agreement_with_each_coder",
    "compare_to_human_reliability",
    "confidence_sweep",
    "ceiling_analysis",
    "Workspace",
    "load_workspace",
    "make_system_prompt",
]

__version__ = "1.0.0"
