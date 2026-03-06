"""
Gemini API synthetic dataset generator.

Calls Gemini with a parameterized prompt and parses the response into
validated ChatML JSONL examples for fine-tuning.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from google import genai
from google.genai import types
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.generators.dataset_validator import validate_example
from src.generators.prompt_templates import build_dataset_prompt

console = Console()


class GeminiDatasetGenerator:
    """Generates synthetic ChatML datasets using Gemini API."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        self.client = genai.Client(api_key=api_key)
        self.model = model

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

        Returns validated examples only. Invalid ones are logged and skipped.
        """
        system_prompt, user_prompt = build_dataset_prompt(
            obiettivo=obiettivo,
            lingua=lingua,
            dominio=dominio,
            n_examples=n_examples,
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=8192,
                response_mime_type="application/json",
            ),
        )

        try:
            raw_examples = json.loads(response.text)
        except json.JSONDecodeError as e:
            console.print(f"[red]JSON parse error from Gemini: {e}[/red]")
            return []

        if not isinstance(raw_examples, list):
            console.print("[red]Gemini response is not a JSON array[/red]")
            return []

        valid_examples = []
        skipped = 0
        for ex in raw_examples:
            is_valid, error = validate_example(ex)
            if is_valid:
                valid_examples.append(ex)
            else:
                skipped += 1
                console.print(f"[yellow]Skipping invalid example: {error}[/yellow]")

        if skipped:
            console.print(f"[yellow]Skipped {skipped} invalid examples[/yellow]")

        return valid_examples

    def generate_to_file(
        self,
        output_path: Path,
        obiettivo: str,
        lingua: str,
        dominio: str,
        n_examples: int = 150,
        batches: int = 3,
        temperature: float = 1.0,
    ) -> tuple[Path, int]:
        """
        Generates the full dataset in batches and writes to a JSONL file.

        Batching avoids Gemini output token limits for large datasets.

        Returns:
            (output_path, total_examples_written)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        examples_per_batch = max(1, n_examples // batches)
        total_written = 0

        with (
            open(output_path, "w", encoding="utf-8") as f,
            Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress,
        ):
            task = progress.add_task(
                f"Generating {n_examples} examples in {batches} batches...",
                total=batches,
            )

            for batch_idx in range(batches):
                progress.update(
                    task,
                    description=f"Batch {batch_idx + 1}/{batches} — calling Gemini...",
                )
                examples = self.generate_batch(
                    obiettivo=obiettivo,
                    lingua=lingua,
                    dominio=dominio,
                    n_examples=examples_per_batch,
                    temperature=temperature,
                )
                for ex in examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    total_written += 1

                progress.advance(task)
                console.print(
                    f"[green]Batch {batch_idx + 1}: {len(examples)} examples written[/green]"
                )

        console.print(
            f"[bold green]Dataset complete: {total_written} examples → {output_path}[/bold green]"
        )
        return output_path, total_written
