"""
fine-tune-forge CLI entry point.

All commands are orchestrated here. No business logic — delegates
to generators, trainers, and exporters.

Usage:
    forge run --dominio restaurant_booking --modello Qwen/Qwen3-0.6B ...
    forge dataset --dominio restaurant_booking
    forge train --dataset data/processed/x.jsonl --dominio restaurant_booking
    forge export --checkpoint outputs/checkpoints/run-xyz/ --format gguf
    forge list-domains
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

app = typer.Typer(
    name="forge",
    help="fine-tune-forge: QLoRA fine-tuning pipeline for vertical LLM agents",
    add_completion=False,
)
console = Console()

CONFIG_DIR = Path("config")
DATA_DIR = Path("data")
OUTPUTS_DIR = Path("outputs")


def _model_key_from_hf_id(hf_model_id: str) -> str | None:
    """Tries to find a matching model config key from the HF model ID."""
    mapping = {
        "Qwen/Qwen3.5-0.8B": "qwen3_5_0_8b",
        "Qwen/Qwen3-0.6B": "qwen3_0_6b",
        "Qwen/Qwen3-4B": "qwen3_4b",
    }
    return mapping.get(hf_model_id)


@app.command()
def run(
    dominio: str = typer.Option(..., "--dominio", "-d", help="Domain config key (e.g. restaurant_booking)"),
    modello: str = typer.Option("Qwen/Qwen3-0.6B", "--modello", "-m", help="HuggingFace model ID"),
    obiettivo: Optional[str] = typer.Option(None, "--obiettivo", help="Override domain obiettivo"),
    lingua: Optional[str] = typer.Option(None, "--lingua", help="Override domain lingua"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Custom run identifier"),
    generator: Optional[str] = typer.Option(None, "--generator", "-g", help="Generator: groq, gemini, ollama (overrides config)"),
    skip_dataset: bool = typer.Option(False, "--skip-dataset", help="Skip dataset generation (use existing)"),
    skip_training: bool = typer.Option(False, "--skip-training", help="Skip training"),
    skip_export: bool = typer.Option(False, "--skip-export", help="Skip export"),
) -> None:
    """Run the full pipeline: dataset generation → training → export."""
    from src.trainers.config_loader import load_config

    run_id = run_id or f"{dominio}-{uuid.uuid4().hex[:8]}"
    model_key = _model_key_from_hf_id(modello)

    overrides = {}
    if obiettivo:
        overrides["obiettivo"] = obiettivo
    if lingua:
        overrides["lingua"] = lingua
    if generator:
        overrides["generator"] = generator
    overrides["hf_model_id"] = modello

    config = load_config(dominio, model_key=model_key, config_dir=CONFIG_DIR, overrides=overrides)

    console.print(f"\n[bold]fine-tune-forge[/bold] — run: [cyan]{run_id}[/cyan]")
    console.print(f"Domain: {dominio} | Model: {modello} | Lingua: {config.lingua}\n")

    dataset_path = DATA_DIR / "processed" / f"{dominio}.jsonl"

    # Step 1: Dataset generation
    if not skip_dataset:
        _run_dataset(config, dataset_path)
    else:
        # Dataset is only needed if training will run
        if not skip_training and not dataset_path.exists():
            console.print(f"[red]Dataset not found: {dataset_path}[/red]")
            raise typer.Exit(1)
        console.print(f"[yellow]Skipping dataset generation, using: {dataset_path}[/yellow]")

    # Step 2: Training
    merged_path = None
    if not skip_training:
        merged_path = _run_training(config, dataset_path, run_id)
    else:
        candidate = OUTPUTS_DIR / "checkpoints" / run_id / "merged"
        if candidate.exists():
            merged_path = candidate
            console.print(f"[yellow]Skipping training, using existing model: {merged_path}[/yellow]")
        else:
            console.print(f"[yellow]Skipping training — no merged model found at {candidate}, skipping export too.[/yellow]")

    # Step 3: Export
    if not skip_export and merged_path is not None:
        _run_export(config, merged_path, run_id)

    console.print(f"\n[bold green]Pipeline complete! Run ID: {run_id}[/bold green]")


@app.command()
def dataset(
    dominio: str = typer.Option(..., "--dominio", "-d", help="Domain config key"),
    output: Optional[str] = typer.Option(None, "--output", help="Output JSONL path"),
    n_examples: Optional[int] = typer.Option(None, "--n-examples", help="Number of examples to generate"),
    generator: Optional[str] = typer.Option(None, "--generator", "-g", help="Generator: groq, gemini, ollama (overrides config)"),
) -> None:
    """Generate a synthetic dataset for a domain (groq/gemini/ollama)."""
    from src.trainers.config_loader import load_config

    config = load_config(dominio, config_dir=CONFIG_DIR)
    if n_examples:
        config.n_examples = n_examples
    if generator:
        config.generator = generator

    output_path = Path(output) if output else DATA_DIR / "processed" / f"{dominio}.jsonl"
    _run_dataset(config, output_path)


@app.command()
def train(
    dataset_path: str = typer.Argument(..., help="Path to the JSONL dataset"),
    dominio: str = typer.Option(..., "--dominio", "-d", help="Domain config key"),
    modello: str = typer.Option("Qwen/Qwen3-0.6B", "--modello", "-m", help="HuggingFace model ID"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Custom run identifier"),
) -> None:
    """Run QLoRA training on an existing dataset."""
    from src.trainers.config_loader import load_config
    from src.trainers.qlora_trainer import run_training

    model_key = _model_key_from_hf_id(modello)
    config = load_config(dominio, model_key=model_key, config_dir=CONFIG_DIR, overrides={"hf_model_id": modello})
    run_id = run_id or f"{dominio}-{uuid.uuid4().hex[:8]}"

    run_training(Path(dataset_path), config, run_id)


@app.command()
def export(
    checkpoint: str = typer.Argument(..., help="Path to merged model checkpoint"),
    format: str = typer.Option("gguf", "--format", "-f", help="Export format: gguf, ollama, hf"),
    name: Optional[str] = typer.Option(None, "--name", help="Model name (for ollama)"),
    repo_id: Optional[str] = typer.Option(None, "--repo-id", help="HuggingFace repo ID (for hf)"),
    system_prompt: Optional[str] = typer.Option(None, "--system-prompt", help="System prompt (for ollama)"),
) -> None:
    """Export a merged model to GGUF, Ollama, or HuggingFace Hub."""
    merged_path = Path(checkpoint)
    output_dir = merged_path.parent.parent / "models" / merged_path.parent.name

    if format == "gguf":
        from src.exporters.gguf_exporter import export_gguf
        export_gguf(merged_path, output_dir)

    elif format == "ollama":
        from src.exporters.gguf_exporter import export_gguf
        from src.exporters.ollama_exporter import export_ollama

        gguf_path = output_dir / "model.gguf"
        if not gguf_path.exists():
            console.print("GGUF not found, generating first...")
            export_gguf(merged_path, output_dir)

        model_name = name or "fine-tuned-agent"
        sp = system_prompt or "You are a helpful assistant."
        export_ollama(gguf_path, model_name, sp, output_dir)

    elif format == "hf":
        from src.exporters.hf_exporter import export_to_hub
        if not repo_id:
            console.print("[red]--repo-id required for hf format[/red]")
            raise typer.Exit(1)
        export_to_hub(merged_path, repo_id)

    else:
        console.print(f"[red]Unknown format: {format}. Use: gguf, ollama, hf[/red]")
        raise typer.Exit(1)


@app.command(name="list-domains")
def list_domains() -> None:
    """List available domain configurations."""
    domains_dir = CONFIG_DIR / "domains"
    if not domains_dir.exists():
        console.print("[red]config/domains/ not found[/red]")
        raise typer.Exit(1)

    table = Table(title="Available Domains")
    table.add_column("Domain Key", style="cyan")
    table.add_column("Lingua")
    table.add_column("Examples")
    table.add_column("Description")

    import yaml
    for yaml_file in sorted(domains_dir.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        obiettivo = (data.get("obiettivo") or data.get("dataset", {}).get("obiettivo") or "").strip().replace("\n", " ")[:60]
        table.add_row(
            yaml_file.stem,
            data.get("lingua", "—"),
            str(data.get("dataset", {}).get("n_examples", "—")),
            obiettivo + ("..." if len(obiettivo) >= 60 else ""),
        )

    console.print(table)


# ── helpers ──────────────────────────────────────────────────────────────────

def _run_dataset(config, output_path: Path) -> None:
    from src.generators.factory import create_generator

    generator_type = getattr(config, "generator", "groq")
    # Pick the right model for the selected generator
    model_for_generator = {
        "groq": getattr(config, "groq_model", None),
        "gemini": getattr(config, "gemini_model", None),
        "ollama": getattr(config, "ollama_generator_model", None),
    }.get(generator_type)

    console.print(f"\n[bold]Step 1: Dataset generation[/bold] → {output_path}")
    console.print(f"Generator: [cyan]{generator_type}[/cyan]")
    gen = create_generator(generator_type, model=model_for_generator)
    _, total = gen.generate_to_file(
        output_path=output_path,
        obiettivo=config.obiettivo,
        lingua=config.lingua,
        dominio=config.domain,
        n_examples=config.n_examples,
        batches=config.batches,
        temperature=config.temperature,
    )
    console.print(f"Dataset: {total} examples\n")


def _run_training(config, dataset_path: Path, run_id: str) -> Path:
    from src.trainers.qlora_trainer import run_training

    console.print(f"[bold]Step 2: QLoRA Training[/bold] — {config.hf_model_id}")
    merged_path = run_training(dataset_path, config, run_id)
    console.print(f"Merged model: {merged_path}\n")
    return merged_path


def _run_export(config, merged_path: Path, run_id: str) -> None:
    output_dir = OUTPUTS_DIR / "models" / run_id
    console.print(f"[bold]Step 3: Export[/bold] → {output_dir}")

    if config.export_gguf:
        from src.exporters.gguf_exporter import export_gguf
        try:
            export_gguf(merged_path, output_dir)
        except FileNotFoundError as e:
            console.print(f"[yellow]GGUF skipped: {e}[/yellow]")

    if config.export_ollama and config.export_gguf:
        from src.exporters.ollama_exporter import export_ollama
        gguf_path = output_dir / "model.gguf"
        if gguf_path.exists():
            model_name = config.ollama_model_name or f"forge-{config.domain}"
            export_ollama(gguf_path, model_name, config.obiettivo, output_dir)

    if config.export_hf_hub and config.hf_repo_id:
        from src.exporters.hf_exporter import export_to_hub
        export_to_hub(merged_path, config.hf_repo_id)


if __name__ == "__main__":
    app()
