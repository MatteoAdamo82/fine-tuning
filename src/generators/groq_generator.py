"""
Groq API synthetic dataset generator.

Uses Groq's fast inference (Llama 3.3 70B) with JSON mode.
Free tier: 14,400 req/day, 30 req/min — much more reliable than Gemini free tier.

Groq requires JSON object mode (not bare arrays), so build_dataset_prompt is
called with wrap_in_object=True and the response is unwrapped via
BaseDatasetGenerator._parse_and_validate().
"""
from __future__ import annotations

import os
import time

from rich.console import Console

from src.generators.base_generator import BaseDatasetGenerator
from src.generators.prompt_templates import build_dataset_prompt

console = Console()


class GroqDatasetGenerator(BaseDatasetGenerator):
    """Generates synthetic ChatML datasets using Groq API."""

    # Groq is fast — short pause is enough between batches
    inter_batch_pause = 3.0

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        try:
            from groq import Groq
        except ImportError as exc:
            raise ImportError(
                "groq package not installed. Run: pip install groq"
            ) from exc

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")

        self.client = Groq(api_key=api_key)
        self.model = model

    def _call_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_retries: int = 5,
    ) -> str:
        """Calls Groq with exponential backoff on rate limit errors (429)."""
        delay = 15
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=8192,
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content
            except Exception as e:
                status = getattr(e, "status_code", None)
                if status == 429 and attempt < max_retries - 1:
                    console.print(
                        f"[yellow]Rate limit (429), waiting {delay}s "
                        f"(attempt {attempt + 1}/{max_retries})...[/yellow]"
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 120)
                else:
                    raise

    def generate_batch(
        self,
        obiettivo: str,
        lingua: str,
        dominio: str,
        n_examples: int,
        temperature: float = 1.0,
    ) -> list[dict]:
        """
        Calls Groq once to generate n_examples ChatML conversations.

        Uses wrap_in_object=True because Groq JSON mode enforces a JSON object
        (not a bare array). The _parse_and_validate helper unwraps it.
        """
        system_prompt, user_prompt = build_dataset_prompt(
            obiettivo=obiettivo,
            lingua=lingua,
            dominio=dominio,
            n_examples=n_examples,
            wrap_in_object=True,
        )
        raw_text = self._call_with_retry(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )
        return self._parse_and_validate(raw_text, source="Groq")
