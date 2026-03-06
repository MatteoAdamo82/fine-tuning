"""
Abstract base class for synthetic dataset generators.

Subclasses implement generate_batch(); this class provides
the shared generate_to_file() batching logic so it is not duplicated.
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.generators.dataset_validator import validate_example

console = Console()


class BaseDatasetGenerator(ABC):
    """
    Abstract generator — subclasses implement generate_batch().

    generate_to_file() is shared: it calls generate_batch() in a loop,
    validates each example, and writes valid ones to a JSONL file.
    """

    # Pause between batches in seconds. Override in subclasses.
    inter_batch_pause: float = 5.0

    @abstractmethod
    def generate_batch(
        self,
        obiettivo: str,
        lingua: str,
        dominio: str,
        n_examples: int,
        temperature: float = 1.0,
    ) -> list[dict]:
        """
        Calls the underlying LLM to generate n_examples ChatML conversations.

        Returns validated examples only. Invalid ones must be logged and skipped
        inside the implementation (not raised).
        """
        ...

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

        Batching avoids output token limits for large datasets.

        Returns:
            (output_path, total_examples_written)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        examples_per_batch = max(1, n_examples // batches)
        total_written = 0
        generator_name = self.__class__.__name__.replace("DatasetGenerator", "")

        with (
            open(output_path, "w", encoding="utf-8") as f,
            Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress,
        ):
            task = progress.add_task(
                f"Generating {n_examples} examples via {generator_name} in {batches} batches...",
                total=batches,
            )

            for batch_idx in range(batches):
                progress.update(
                    task,
                    description=(
                        f"Batch {batch_idx + 1}/{batches} — calling {generator_name}..."
                    ),
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

                if batch_idx < batches - 1:
                    time.sleep(self.inter_batch_pause)

        console.print(
            f"[bold green]Dataset complete: {total_written} examples → {output_path}[/bold green]"
        )
        return output_path, total_written

    # ── helpers shared by subclasses ────────────────────────────────────────

    @staticmethod
    def _parse_and_validate(raw_text: str, source: str) -> list[dict]:
        """
        Parses JSON text and validates each example.

        Handles both bare arrays and wrapper objects {"examples": [...]}
        because different APIs enforce different JSON mode constraints.

        Returns only valid examples.
        """
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as e:
            console.print(f"[red]JSON parse error from {source}: {e}[/red]")
            return []

        # Unwrap {"examples": [...]} or any {"key": [...]} wrapper
        if isinstance(parsed, dict):
            raw_examples = None
            for v in parsed.values():
                if isinstance(v, list):
                    raw_examples = v
                    break
            if raw_examples is None:
                console.print(f"[red]{source}: no list found in JSON object[/red]")
                return []
        elif isinstance(parsed, list):
            raw_examples = parsed
        else:
            console.print(f"[red]{source}: unexpected JSON structure[/red]")
            return []

        valid_examples: list[dict] = []
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
