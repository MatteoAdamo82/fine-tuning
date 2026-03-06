"""
HuggingFace Hub exporter.

Pushes the merged model and tokenizer to HuggingFace Hub.
Requires HF_TOKEN env var with write access.
"""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, login
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()


def export_to_hub(
    merged_model_path: Path,
    repo_id: str,
    private: bool = True,
    commit_message: str = "Fine-tuned with fine-tune-forge",
) -> str:
    """
    Pushes the merged model to HuggingFace Hub.

    Args:
        merged_model_path: path to the merged model directory
        repo_id: HuggingFace repo ID (e.g. 'username/agente-ristorante')
        private: whether the repo should be private
        commit_message: commit message for the push

    Returns:
        URL of the uploaded model

    Raises:
        ValueError: if HF_TOKEN is not set
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")

    login(token=hf_token)

    console.print(f"Loading model from {merged_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(merged_model_path))
    model = AutoModelForCausalLM.from_pretrained(str(merged_model_path))

    console.print(f"Pushing to HuggingFace Hub: {repo_id} (private={private})...")
    model.push_to_hub(
        repo_id,
        private=private,
        commit_message=commit_message,
        token=hf_token,
    )
    tokenizer.push_to_hub(
        repo_id,
        private=private,
        commit_message=commit_message,
        token=hf_token,
    )

    url = f"https://huggingface.co/{repo_id}"
    console.print(f"[bold green]Model pushed: {url}[/bold green]")
    return url
