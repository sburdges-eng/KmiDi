#!/usr/bin/env python3
"""
Vertex AI training entrypoint for Neural Language Model (NLM) Fine-Tuning.

Expects Vertex AI contract:
  - Hyperparameters: Passed as command-line arguments.
  - Input data: GCS paths passed as arguments or via AIP_ env vars.
  - Output: AIP_MODEL_DIR (Vertex AI uses this to save the model to GCS).

Usage inside container:
  python vertex_train_nlm.py --base_model meta-llama/Llama-3-8B-Instruct ...
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Repo root on host; in Vertex container we rely on installed package
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vertex_train_nlm")

# Vertex AI paths
AIP_MODEL_DIR = os.environ.get("AIP_MODEL_DIR", "/tmp/kmidi_model") # Fallback to /tmp for local testing

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    import datasets
except ImportError as e:
    logger.error("Missing NLM training dependencies. Did you install `[nlm_train]`?")
    raise e


def resolve_gcs_path(path_str: str) -> str:
    """Resolve gs:// to Vertex AI GCS Fuse path."""
    if path_str.startswith("gs://"):
        bucket_name = path_str.replace("gs://", "").split("/")[0]
        relative_path = "/".join(path_str.replace("gs://", "").split("/")[1:])
        return f"/gcs/{bucket_name}/{relative_path}"
    return path_str


def load_dataset(dataset_path: str):
    """Load dataset from huggingface or jsonl files on GCS."""
    resolved_path = resolve_gcs_path(dataset_path)
    
    if os.path.isdir(resolved_path) or os.path.isfile(resolved_path):
        # Local or GCS fuse file
        if resolved_path.endswith(".jsonl"):
            return datasets.load_dataset("json", data_files=resolved_path, split="train")
        else:
            return datasets.load_from_disk(resolved_path)
    else:
        # Assume HuggingFace dataset
        return datasets.load_dataset(dataset_path, split="train")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vertex AI NLM/LLM training")
    
    # Generic training args
    parser.add_argument("--base_model", type=str, required=True, help="HuggingFace Model ID or GCS Path")
    parser.add_argument("--dataset_path", type=str, required=True, help="GCS path or HF dataset ID")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""), help="HF token for gated models")
    
    # LoRA specific
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    args = parser.parse_args()

    logger.info(f"Vertex AI NLM training starting for model: {args.base_model}")

    # Resolve output dir
    model_dir = resolve_gcs_path(AIP_MODEL_DIR)
    os.makedirs(model_dir, exist_ok=True)
    
    # Resolving Base Model GCS Path if specified via gs:// 
    # (if you staged it to GCS or are pointing to HF)
    model_loaded_path = resolve_gcs_path(args.base_model)
    
    logger.info("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_loaded_path, 
        token=args.hf_token,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading Model (4-bit)...")
    # For bitsandbytes quantization (QLoRA)
    bnb_config = None
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    except Exception as e:
        logger.warning(f"Failed to setup 4-bit quantization, proceeding without: {e}")

    model = AutoModelForCausalLM.from_pretrained(
        model_loaded_path,
        quantization_config=bnb_config,
        device_map="auto",
        token=args.hf_token,
        trust_remote_code=True,
    )
    model.config.use_cache = False # Required for gradient checkpointing
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    # Setup LoRA
    logger.info("Setting up LoRA configuration...")
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_rank,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load Dataset
    logger.info(f"Loading dataset from: {args.dataset_path}")
    dataset = load_dataset(args.dataset_path)

    # Setup Trainer
    training_args = TrainingArguments(
        output_dir=os.path.join(model_dir, "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=10,
        learning_rate=args.learning_rate,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        num_train_epochs=args.epochs,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard"
    )

    logger.info("Initializing SFTTrainer...")
    # Assumes dataset has a 'text' column for standard causal LM objective 
    # Or format instruction using dataset_text_field
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    logger.info("Starting Training...")
    trainer.train()

    logger.info("Saving final adapter weights...")
    final_output_path = os.path.join(model_dir, "final_adapter")
    trainer.model.save_pretrained(final_output_path)
    tokenizer.save_pretrained(final_output_path)

    logger.info(f"Training complete. Artifacts saved to: {model_dir}")


if __name__ == "__main__":
    main()
