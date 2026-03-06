"""
Loads and merges YAML configs into a typed TrainingConfig dataclass.

Merges: base.yaml <- domain config <- model config <- CLI overrides.
Later values override earlier ones.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainingConfig:
    # Model
    hf_model_id: str = "Qwen/Qwen3-0.6B"
    ollama_tag: str = "qwen3:0.6b"
    architecture: str = "qwen3"
    model_class: str = "AutoModelForCausalLM"

    # Domain / dataset
    domain: str = ""
    lingua: str = "Italiano"
    obiettivo: str = ""
    n_examples: int = 150
    batches: int = 3
    generator: str = "groq"
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_model: str = "gemini-2.0-flash"
    ollama_generator_model: str = "qwen3:4b"
    temperature: float = 1.0

    # Training hyperparams
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 1024
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    save_steps: int = 50
    logging_steps: int = 10
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]
    )

    # Export
    export_gguf: bool = True
    export_hf_hub: bool = False
    export_ollama: bool = False
    ollama_model_name: str = ""
    hf_repo_id: str = ""

    # Runtime
    force_cpu: bool = False
    output_dir: str = "outputs"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merges override into base recursively."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_to_config(config: TrainingConfig, data: dict) -> None:
    """Flattens nested YAML dict and sets fields on the dataclass."""
    flat: dict[str, Any] = {}

    # Flatten training: subtree
    for k, v in data.get("training", {}).items():
        flat[k] = v

    # Flatten dataset: subtree
    for k, v in data.get("dataset", {}).items():
        flat[k] = v

    # Flatten model: subtree
    model_data = data.get("model", {})
    if "hf_id" in model_data:
        flat["hf_model_id"] = model_data["hf_id"]
    if "ollama_tag" in model_data:
        flat["ollama_tag"] = model_data["ollama_tag"]
    if "architecture" in model_data:
        flat["architecture"] = model_data["architecture"]
    if "model_class" in model_data:
        flat["model_class"] = model_data["model_class"]

    # Flatten export: subtree
    export_data = data.get("export", {})
    if "gguf" in export_data:
        flat["export_gguf"] = export_data["gguf"]
    if "hf_hub" in export_data:
        flat["export_hf_hub"] = export_data["hf_hub"]
    if "ollama" in export_data:
        flat["export_ollama"] = export_data["ollama"]
    if "ollama_model_name" in export_data:
        flat["ollama_model_name"] = export_data["ollama_model_name"]
    if "hf_repo_id" in export_data:
        flat["hf_repo_id"] = export_data["hf_repo_id"]

    # Top-level domain fields
    for k in ("domain", "lingua", "obiettivo"):
        if k in data:
            flat[k] = data[k]

    # Apply flat dict to dataclass
    for key, value in flat.items():
        if hasattr(config, key):
            setattr(config, key, value)


def load_config(
    domain: str,
    model_key: str | None = None,
    config_dir: Path | None = None,
    overrides: dict | None = None,
) -> TrainingConfig:
    """
    Loads and merges configs for the given domain and optional model key.

    Load order (later overrides earlier):
    1. base.yaml defaults
    2. config/domains/{domain}.yaml
    3. config/models/{model_key}.yaml (if provided)
    4. overrides dict (CLI flags)
    5. FORCE_CPU env var

    Args:
        domain: domain name (e.g. 'restaurant_booking')
        model_key: model config key (e.g. 'qwen3_0_6b'), optional
        config_dir: path to config/ directory, defaults to ./config
        overrides: dict of field overrides from CLI
    """
    if config_dir is None:
        config_dir = Path("config")

    base_data = _load_yaml(config_dir / "base.yaml")
    domain_data = _load_yaml(config_dir / "domains" / f"{domain}.yaml")
    model_data = _load_yaml(config_dir / "models" / f"{model_key}.yaml") if model_key else {}

    merged = _deep_merge(_deep_merge(base_data, domain_data), model_data)

    config = TrainingConfig()
    _apply_to_config(config, merged)

    if overrides:
        for key, value in overrides.items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)

    config.force_cpu = os.environ.get("FORCE_CPU", "").lower() in ("true", "1", "yes")
    config.domain = domain

    return config
