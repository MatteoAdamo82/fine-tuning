"""
QLoRA fine-tuning orchestrator using TRL SFTTrainer.

Uses BitsAndBytesConfig for 4-bit quantization and LoraConfig for
parameter-efficient fine-tuning. Designed for single-GPU or CPU execution.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from rich.console import Console
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

from src.trainers.config_loader import TrainingConfig

console = Console()


def _get_bnb_config(force_cpu: bool) -> BitsAndBytesConfig | None:
    """Returns 4-bit quantization config, or None for CPU training."""
    if force_cpu or not torch.cuda.is_available():
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def _load_jsonl_dataset(dataset_path: Path) -> Dataset:
    """Loads a JSONL file as a HuggingFace Dataset."""
    ds = load_dataset("json", data_files=str(dataset_path), split="train")
    return ds


def run_training(
    dataset_path: Path,
    config: TrainingConfig,
    run_id: str,
) -> Path:
    """
    Runs QLoRA fine-tuning and saves the merged model.

    Args:
        dataset_path: path to the validated JSONL dataset
        config: merged TrainingConfig
        run_id: unique identifier for this training run

    Returns:
        Path to the saved merged model directory
    """
    output_dir = Path(config.output_dir) / "checkpoints" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    force_cpu = config.force_cpu or not torch.cuda.is_available()

    if force_cpu:
        console.print("[yellow]FORCE_CPU mode: training on CPU (slow)[/yellow]")
    else:
        console.print(f"[green]GPU detected: {torch.cuda.get_device_name(0)}[/green]")

    # Load tokenizer
    console.print(f"Loading tokenizer: {config.hf_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.hf_model_id,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    console.print(f"Loading model: {config.hf_model_id}")
    bnb_config = _get_bnb_config(force_cpu)
    model_kwargs: dict = {
        "trust_remote_code": True,
        "token": os.environ.get("HF_TOKEN"),
    }
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float32

    # Use AutoModelForImageTextToText for VLM architectures (e.g. qwen3_5)
    is_vlm = getattr(config, "model_class", "") == "AutoModelForImageTextToText"
    if is_vlm:
        model = AutoModelForImageTextToText.from_pretrained(config.hf_model_id, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(config.hf_model_id, **model_kwargs)

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        target_modules=config.lora_target_modules,
    )

    # Load dataset
    console.print(f"Loading dataset: {dataset_path}")
    dataset = _load_jsonl_dataset(dataset_path)
    console.print(f"Dataset size: {len(dataset)} examples")

    # Training arguments
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        save_total_limit=2,
        fp16=not force_cpu and torch.cuda.is_available(),
        bf16=False,
        dataloader_num_workers=0,
        report_to="none",
        max_seq_length=config.max_seq_length,
    )

    # SFTTrainer handles ChatML formatting via tokenizer.apply_chat_template
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    console.print("[bold]Starting training...[/bold]")
    trainer.train()

    # Save adapter
    adapter_path = output_dir / "adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    console.print(f"[green]Adapter saved: {adapter_path}[/green]")

    # Merge adapter into base model and save
    merged_path = output_dir / "merged"
    console.print("Merging adapter into base model...")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))
    console.print(f"[bold green]Merged model saved: {merged_path}[/bold green]")

    return merged_path
