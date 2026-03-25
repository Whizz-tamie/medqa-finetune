import sys
import argparse
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
)
from trl import SFTTrainer, SFTConfig

# Global Configurations
CONFIG = {
    "model_id": "Qwen/Qwen2.5-7B-Instruct",
    "train_path": "data/train.jsonl",
    "val_path": "data/val.jsonl",
    "output_dir": "./outputs/qwen-medqa-adapter",
    "max_length": 512,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "learning_rate": 5e-5,
    "epochs": 3,
    "batch_size": 8,
    "grad_accum": 4,
}

TEST_CONFIG_OVERRIDES = {
    "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
    "max_length": 256,
    "batch_size": 1,
    "grad_accum": 1,
    "learning_rate": 1e-5,
}

# Model Setup
def get_runtime_settings():
    if torch.cuda.is_available():
        return {
            "device_name": torch.cuda.get_device_name(0),
            "torch_dtype": torch.bfloat16,
            "bf16": True,
            "optim": "adamw_torch_fused",
            "device_map": "auto",
            "target_device": None,
        }

    if torch.backends.mps.is_available():
        return {
            "device_name": "MPS",
            "torch_dtype": torch.float32,
            "bf16": False,
            "optim": "adamw_torch",
            "device_map": None,
            "target_device": "mps",
        }

    return {
        "device_name": "CPU",
        "torch_dtype": torch.float32,
        "bf16": False,
        "optim": "adamw_torch",
        "device_map": None,
        "target_device": "cpu",
    }


def get_model(model_id, runtime_settings):
    print(f"[INFO] Mode: Standard LoRA ({runtime_settings['torch_dtype']})")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=runtime_settings["torch_dtype"],
        device_map=runtime_settings["device_map"],
        trust_remote_code=True,
    )

    if runtime_settings["target_device"] is not None:
        model = model.to(runtime_settings["target_device"])

    # disable cache for training
    model.config.use_cache = False

    # helpful for memory
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    return model, peft_config

# Training pipeline
def run_training(is_test=False):
    run_config = {**CONFIG, **TEST_CONFIG_OVERRIDES} if is_test else CONFIG
    runtime_settings = get_runtime_settings()

    tokenizer = AutoTokenizer.from_pretrained(run_config["model_id"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model, peft_config = get_model(run_config["model_id"], runtime_settings)

    dataset = load_dataset("json", data_files={
    "train": CONFIG["train_path"], 
    "validation": CONFIG["val_path"]
    })

    expected_columns = {"prompt", "completion"}
    assert expected_columns.issubset(dataset["train"].features), "Dataset must contain 'prompt' and 'completion'"
    
    train_set = dataset["train"].select(range(min(20, len(dataset["train"])))) if is_test else dataset["train"]
    val_set = dataset["validation"].select(range(min(10, len(dataset["validation"])))) if is_test else dataset["validation"]

    sft_config = SFTConfig(
        # --- Identification & Reporting ---
        project="medqa-qwen-finetune",
        run_name="qwen-medqa-lora",
        report_to="none" if is_test else "wandb",
        output_dir=CONFIG["output_dir"],

        # --- Training Hyperparameters ---
        per_device_train_batch_size=run_config["batch_size"],
        gradient_accumulation_steps=run_config["grad_accum"],
        learning_rate=run_config["learning_rate"],
        num_train_epochs=run_config["epochs"],
        lr_scheduler_type="cosine",
        optim=runtime_settings["optim"],
        bf16=runtime_settings["bf16"],

        # --- Evaluation & Saving ---
        eval_strategy="steps",
        eval_steps=1 if is_test else 100,
        save_strategy="no" if is_test else "steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=not is_test,
        metric_for_best_model="eval_loss",

        # --- SFT Specific Logic
        max_length=run_config["max_length"],
        packing=False,
        completion_only_loss=True,
        dataset_text_field=None,
        
        # --- Operational ---
        max_steps=5 if is_test else -1,
        remove_unused_columns=False, # Important for PEFT/SFT stability
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_set,
        eval_dataset=val_set,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_config
    )

    if is_test:
        print(
            "[INFO] Smoke test config: "
            f"model={run_config['model_id']}, "
            f"batch_size={run_config['batch_size']}, "
            f"grad_accum={run_config['grad_accum']}, "
            f"max_length={run_config['max_length']}, "
            f"lr={run_config['learning_rate']}"
        )
    print(
        f"[STATUS] Launching {'SMOKE TEST' if is_test else 'FULL TRAINING'} "
        f"with {run_config['model_id']} "
        f"on {runtime_settings['device_name']} "
        f"(dtype={runtime_settings['torch_dtype']}, optim={runtime_settings['optim']})"
    )
    trainer.train()

    # Only save the model if we are NOT in a test run
    if not is_test:
        trainer.save_model(CONFIG["output_dir"])
        print(f"[SUCCESS] Adapter saved to {CONFIG['output_dir']}")
    else:
        print("[INFO] Smoke test complete. No model was saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run in smoke test mode")
    args, _ = parser.parse_known_args()
    
    run_training(is_test=args.test)
