"""
HuggingFace Hub exporter.

Pushes the merged model and tokenizer to HuggingFace Hub.
Requires HF_TOKEN env var with write access.

Token permissions:
    Go to https://huggingface.co/settings/tokens → New token → Type: Write
    (or fine-grained with "Create/Write repositories" permission)
"""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()


def _validate_token(api: HfApi, repo_id: str) -> str:
    """Validates token and checks namespace ownership. Returns the token username."""
    try:
        info = api.whoami()
    except HfHubHTTPError as e:
        raise ValueError(
            "HF_TOKEN non valido o scaduto.\n"
            "  → Vai su https://huggingface.co/settings/tokens e crea un nuovo token."
        ) from e

    token_user = info.get("name", "")
    repo_namespace = repo_id.split("/")[0] if "/" in repo_id else ""

    if repo_namespace and repo_namespace.lower() != token_user.lower():
        raise ValueError(
            f"Namespace mismatch: repo_id '{repo_id}' usa '{repo_namespace}', "
            f"ma il token appartiene a '{token_user}'.\n"
            f"  → Usa repo_id '{token_user}/{repo_id.split('/')[-1]}' "
            f"oppure accedi con l'account corretto."
        )
    return token_user


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
        ValueError: if HF_TOKEN is not set, invalid, or has wrong permissions
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN non impostato.\n"
            "  → Crea un token su https://huggingface.co/settings/tokens (tipo: Write)\n"
            "  → Aggiungilo ai Secrets Colab con nome HF_TOKEN"
        )

    api = HfApi(token=hf_token)

    # Validate token and namespace before attempting push
    _validate_token(api, repo_id)

    # Create repo explicitly — gives a clear 403 error if token lacks write access,
    # rather than a cryptic error buried inside push_to_hub internals
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
            token=hf_token,
        )
        console.print(f"[green]Repo ready: {repo_id}[/green]")
    except HfHubHTTPError as e:
        if "403" in str(e):
            raise ValueError(
                f"403 Forbidden: il token non ha permessi di scrittura su '{repo_id}'.\n"
                "  → Vai su https://huggingface.co/settings/tokens\n"
                "  → Crea un token con Type=Write (o fine-grained con 'Write repositories')\n"
                "  → Aggiorna il Secret HF_TOKEN in Colab"
            ) from e
        raise

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
    console.print(f"[bold green]✓ Modello pubblicato: {url}[/bold green]")
    return url
