"""
Ollama local API synthetic dataset generator.

No API key, no rate limits. Requires Ollama running locally.
Uses Python's built-in urllib — no additional dependencies.

Default model: qwen3:4b (good quality + fits 6GB VRAM).
Override via OLLAMA_GENERATOR_MODEL env var.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from rich.console import Console

from src.generators.base_generator import BaseDatasetGenerator
from src.generators.prompt_templates import build_dataset_prompt

console = Console()


class OllamaDatasetGenerator(BaseDatasetGenerator):
    """Generates synthetic ChatML datasets using a local Ollama instance."""

    # Local inference — minimal pause needed between batches
    inter_batch_pause = 0.5

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("OLLAMA_GENERATOR_MODEL", "qwen3:4b")
        self.base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self._check_connection()

    def _check_connection(self) -> None:
        """Verifies Ollama is reachable before generation starts."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5):
                pass
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Is Ollama running? Try: ollama serve"
            ) from e

    def _call_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> str:
        """
        Calls the Ollama /api/chat endpoint with JSON format mode.

        format="json" instructs Ollama to output a valid JSON object.
        Combines system and user messages per Ollama chat API spec.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 8192,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # Long timeout — local inference can be slow for large batches
        with urllib.request.urlopen(req, timeout=600) as response:
            result = json.loads(response.read().decode("utf-8"))

        return result["message"]["content"]

    def generate_batch(
        self,
        obiettivo: str,
        lingua: str,
        dominio: str,
        n_examples: int,
        temperature: float = 1.0,
    ) -> list[dict]:
        """
        Calls local Ollama once to generate n_examples ChatML conversations.

        Uses wrap_in_object=True because Ollama JSON mode enforces a JSON object.
        The _parse_and_validate helper unwraps it.
        """
        system_prompt, user_prompt = build_dataset_prompt(
            obiettivo=obiettivo,
            lingua=lingua,
            dominio=dominio,
            n_examples=n_examples,
            wrap_in_object=True,
        )
        console.print(
            f"[dim]Ollama ({self.model}): generating {n_examples} examples...[/dim]"
        )
        raw_text = self._call_ollama(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )
        return self._parse_and_validate(raw_text, source=f"Ollama/{self.model}")
