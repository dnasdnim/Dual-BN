# phase1_sft.py
"""
DUAL-BN Phase 1: Bidirectional Supervised Fine-Tuning.
QLoRA on DeepSeek-R1-Distill-Qwen-14B, training all four extraction passes
plus the generation direction with token-level cross-entropy.

Run after preparing_training_data.py has produced data/train and data/val.
"""

import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
 
DATA_DIR = "data"
OUTPUT_DIR = "outputs/phase1_q1"

# Training hyperparameters
EPOCHS = 2
PER_DEVICE_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8           # effective batch size = 32
LEARNING_RATE = 2e-4
WARMUP_STEPS = 100
MAX_SEQ_LENGTH = 3072
LR_SCHEDULER = "cosine"

# LoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

HF_TOKEN = os.environ.get("HF_TOKEN", None)


# ─────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────

def setup_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, token=HF_TOKEN, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def setup_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────────────────────
# Data formatting
# ─────────────────────────────────────────────────────────────

def format_dataset(ds, tokenizer):
    """Apply chat template to convert messages -> text string."""
    def _format(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    cols_to_remove = [c for c in ds.column_names if c != "text"]
    return ds.map(_format, batched=True, remove_columns=cols_to_remove)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print(f"=== DUAL-BN Phase 1 SFT ===")
    print(f"Base model: {MODEL}")

    tokenizer = setup_tokenizer()

    print(f"\nLoading datasets from {DATA_DIR}/")
    train_ds = load_from_disk(f"{DATA_DIR}/train")
    val_ds = load_from_disk(f"{DATA_DIR}/val")
    print(f"  Train rows: {len(train_ds)}")
    print(f"  Val rows:   {len(val_ds)}")

    print("\nFormatting with chat template...")
    train_ds = format_dataset(train_ds, tokenizer)
    val_ds = format_dataset(val_ds, tokenizer)

    print("\nLoading model (4-bit + LoRA)...")
    model = setup_model()

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,                  # "outputs/phase1_pi1"
        num_train_epochs=3,                     # real run: 2 epochs
        save_steps=10000,
        per_device_train_batch_size=1,          # least memory (same as smoke)
        per_device_eval_batch_size=1,           # least memory for eval too
        gradient_accumulation_steps=1,         # 1 × 32 = effective batch 32 (free memory-wise)
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        max_grad_norm=1.0,
        logging_steps=20,
        logging_first_step=True,
        max_length=3072,                        # covers all rows (max observed 2776), no truncation
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="text",
        packing=False,
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else ["none"],
        save_strategy="steps",
        # save_steps=10000,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=10000,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        loss_type="nll",                        # silence chunked_nll FutureWarning
        seed=42,
        # --- keep entropy logging OFF to save memory (the OOM source) ---
        # uncomment whichever field your TRL version exposes:
        # compute_entropy=False,
        # log_entropy=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    n_steps = len(train_ds) * EPOCHS // (PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS)
    print(f"\nStarting training...")
    print(f"  Effective batch size: {PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"  Total steps: ~{n_steps}")
    print(f"  Expected wall clock: ~12-16 hours on A100 80GB")

    trainer.train()

    print(f"\nSaving final adapter to {OUTPUT_DIR}/final/")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print("\nPhase 1 complete.")


if __name__ == "__main__":
    main()
