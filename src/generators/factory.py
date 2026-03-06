"""
Factory for dataset generators.

Returns the correct BaseDatasetGenerator subclass based on the
generator type string from config or CLI.

Supported types:
    groq    — Groq API (Llama 3.3 70B), free tier, fast. Requires GROQ_API_KEY.
    gemini  — Google Gemini API, free tier but tight RPM limits. Requires GEMINI_API_KEY.
    ollama  — Local Ollama instance, no rate limits. Requires ollama running.
"""
from __future__ import annotations

from src.generators.base_generator import BaseDatasetGenerator

_SUPPORTED = ("groq", "gemini", "ollama")


def create_generator(
    generator_type: str,
    model: str | None = None,
) -> BaseDatasetGenerator:
    """
    Instantiates and returns the requested dataset generator.

    Args:
        generator_type: one of "groq", "gemini", "ollama"
        model: optional model override (e.g. "llama-3.1-8b-instant" for Groq,
               "gemini-1.5-flash" for Gemini, "qwen3:0.6b" for Ollama)

    Raises:
        ValueError: if generator_type is not supported
        ImportError: if required package is missing (groq)
        ValueError: if required env var is missing (GROQ_API_KEY, GEMINI_API_KEY)
        ConnectionError: if Ollama is not reachable
    """
    gt = generator_type.lower().strip()

    if gt == "groq":
        from src.generators.groq_generator import GroqDatasetGenerator

        kwargs = {}
        if model:
            kwargs["model"] = model
        return GroqDatasetGenerator(**kwargs)

    elif gt == "gemini":
        from src.generators.gemini_generator import GeminiDatasetGenerator

        kwargs = {}
        if model:
            kwargs["model"] = model
        return GeminiDatasetGenerator(**kwargs)

    elif gt == "ollama":
        from src.generators.ollama_generator import OllamaDatasetGenerator

        kwargs = {}
        if model:
            kwargs["model"] = model
        return OllamaDatasetGenerator(**kwargs)

    else:
        raise ValueError(
            f"Unknown generator type: '{generator_type}'. "
            f"Supported: {', '.join(_SUPPORTED)}"
        )
