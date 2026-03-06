"""
Gemini API synthetic dataset generator.

Calls Gemini with a parameterized prompt and parses the response into
validated ChatML JSONL examples for fine-tuning.

Note: Gemini supports bare JSON arrays via response_mime_type="application/json",
so wrap_in_object=False. generate_to_file() is inherited from BaseDatasetGenerator.
"""
from __future__ import annotations

import json
import os
import time

from google import genai
from google.genai import types
from google.genai.errors import ClientError
from rich.console import Console

from src.generators.base_generator import BaseDatasetGenerator
from src.generators.prompt_templates import build_dataset_prompt

console = Console()


class GeminiDatasetGenerator(BaseDatasetGenerator):
    """Generates synthetic ChatML datasets using Gemini API."""

    # Longer pause — Gemini free tier has tight RPM limits
    inter_batch_pause = 10.0

    def __init__(self, model: str = "gemini-2.0-flash"):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _generate_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_retries: int = 5,
    ):
        """Calls Gemini with exponential backoff on 429 rate-limit errors."""
        delay = 30  # seconds — respect free-tier RPM limit
        for attempt in range(max_retries):
            try:
                return self.client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=temperature,
                        max_output_tokens=8192,
                        response_mime_type="application/json",
                    ),
                )
            except ClientError as e:
                if getattr(e, "code", None) == 429 and attempt < max_retries - 1:
                    console.print(
                        f"[yellow]Rate limit hit (429), waiting {delay}s "
                        f"(attempt {attempt + 1}/{max_retries})...[/yellow]"
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 120)  # cap at 2 minutes
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
        Calls Gemini once to generate n_examples ChatML conversations.

        Gemini supports bare JSON arrays via response_mime_type, so
        wrap_in_object=False. _parse_and_validate handles the rest.
        """
        system_prompt, user_prompt = build_dataset_prompt(
            obiettivo=obiettivo,
            lingua=lingua,
            dominio=dominio,
            n_examples=n_examples,
            wrap_in_object=False,
        )
        response = self._generate_with_retry(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )
        return self._parse_and_validate(response.text, source="Gemini")
