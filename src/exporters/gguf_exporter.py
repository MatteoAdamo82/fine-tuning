"""
GGUF exporter using llama.cpp convert_hf_to_gguf.py.

Converts a merged HuggingFace model directory to GGUF format
for use with llama.cpp and Ollama.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console

console = Console()

# Default llama.cpp path — can be overridden via LLAMA_CPP_PATH env var
DEFAULT_LLAMA_CPP_PATH = Path.home() / "llama.cpp"


def _find_convert_script(llama_cpp_path: Path) -> Path | None:
    """Finds the convert_hf_to_gguf.py script in the llama.cpp repo."""
    candidates = [
        llama_cpp_path / "convert_hf_to_gguf.py",
        llama_cpp_path / "convert-hf-to-gguf.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def export_gguf(
    merged_model_path: Path,
    output_dir: Path,
    quantization: str = "q4_k_m",
    llama_cpp_path: Path | None = None,
) -> Path:
    """
    Converts a merged HuggingFace model to GGUF format.

    Args:
        merged_model_path: path to the merged model directory
        output_dir: where to save the .gguf file
        quantization: quantization type (q4_k_m, q8_0, f16, etc.)
        llama_cpp_path: path to llama.cpp repo, defaults to ~/llama.cpp

    Returns:
        Path to the generated .gguf file

    Raises:
        FileNotFoundError: if llama.cpp is not found
        RuntimeError: if conversion fails
    """
    import os
    if llama_cpp_path is None:
        env_path = os.environ.get("LLAMA_CPP_PATH")
        llama_cpp_path = Path(env_path) if env_path else DEFAULT_LLAMA_CPP_PATH

    convert_script = _find_convert_script(llama_cpp_path)
    if convert_script is None:
        raise FileNotFoundError(
            f"llama.cpp convert script not found in {llama_cpp_path}.\n"
            "Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp\n"
            "Or set LLAMA_CPP_PATH env var."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = output_dir / "model.gguf"

    console.print(f"Converting to GGUF ({quantization})...")
    cmd = [
        sys.executable,
        str(convert_script),
        str(merged_model_path),
        "--outfile", str(gguf_path),
        "--outtype", quantization,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]GGUF conversion failed:[/red]\n{result.stderr}")
        raise RuntimeError(f"GGUF conversion failed: {result.stderr}")

    console.print(f"[bold green]GGUF exported: {gguf_path}[/bold green]")
    return gguf_path
