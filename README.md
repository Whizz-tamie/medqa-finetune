# medqa-finetune

Fine-tune `Qwen/Qwen2.5-7B-Instruct` on MedQA USMLE-style multiple-choice questions using LoRA, TRL, and Hugging Face tooling.

## What This Repo Does

This project turns the MedQA dataset into conversational prompt-completion examples and trains a Qwen instruct model to answer with the correct option letter in the format `#### A`.

By default, full training targets `Qwen/Qwen2.5-7B-Instruct`. Smoke tests also use `Qwen/Qwen2.5-7B-Instruct` when CUDA is available, and fall back to `Qwen/Qwen2.5-0.5B-Instruct` on non-CUDA machines.

The current workflow is:

1. Download and preprocess MedQA into JSONL files
2. Run a lightweight local smoke test
3. Launch full LoRA fine-tuning
4. Evaluate either the trained adapter or a standalone model

## Features

- MedQA preprocessing from Hugging Face datasets
- Conversational prompt-completion training format
- LoRA fine-tuning with PEFT
- TRL `SFTTrainer` training loop
- Reusable evaluation script for adapters and standalone models
- Safer `--test` mode for local validation
- Device-aware runtime settings for CUDA and non-CUDA environments

## Quickstart

### Install dependencies

This repo uses `uv` for dependency management.

```bash
uv sync
```

Optional:

```bash
source .venv/bin/activate
```

### Prepare the dataset

```bash
python main.py prepare
```

This creates:

- `data/train.jsonl`
- `data/val.jsonl`
- `data/test.jsonl`

### Run a local smoke test

```bash
python main.py train --test
```

Smoke test mode:

- uses `Qwen/Qwen2.5-7B-Instruct` on CUDA
- falls back to `Qwen/Qwen2.5-0.5B-Instruct` on non-CUDA machines
- uses a small subset of the data
- uses a reduced-memory config
- runs only 5 training steps
- does not save checkpoints or a final model

To use Weights & Biases, authenticate first:

```bash
wandb login
```

Then, to test Weights & Biases logging without launching a full run:

```bash
python main.py train --test --wandb
```

This keeps the smoke-test behavior but enables W&B logging with the fixed run name `qwen-medqa-lora-smoke`.

### Run full training

```bash
python main.py train
```

The LoRA adapter is written to:

```text
outputs/qwen-medqa-adapter
```

### Evaluate a trained adapter

```bash
python main.py eval
```

### Evaluate a standalone model

```bash
python main.py eval --model-path Qwen/Qwen2.5-0.5B-Instruct --max-samples 20
```

### Log evaluation results to W&B

```bash
python main.py eval --wandb
```

Evaluation supports:

- saved LoRA adapters via `--adapter-path`
- standalone models via `--model-path`
- optional W&B logging via `--wandb`

## Dataset Format

Training examples are stored as conversational prompt-completion records:

```json
{
  "prompt": [
    {"role": "system", "content": "You are a medical expert answering USMLE-style questions."},
    {"role": "user", "content": "Question: ..."}
  ],
  "completion": [
    {"role": "assistant", "content": "#### D"}
  ]
}
```

This preserves chat roles while allowing `completion_only_loss=True` during training.

## Training Setup

- Full training model: `Qwen/Qwen2.5-7B-Instruct`
- Smoke test model on CUDA: `Qwen/Qwen2.5-7B-Instruct`
- Smoke test model on non-CUDA: `Qwen/Qwen2.5-0.5B-Instruct`
- Trainer: TRL `SFTTrainer`
- Fine-tuning method: LoRA via PEFT
- Loss setup: completion-only loss
- Default max sequence length: `512`

Runtime behavior:

- CUDA uses `bfloat16` and fused AdamW
- Non-CUDA environments use `float32` and standard AdamW

Local smoke tests are intended for pipeline validation, not for judging final model quality. For reliable training results, use a CUDA machine.

## CLI

The repo uses [`main.py`](/Users/godblessjames/dev/medqa-finetune/main.py) as a small command entry point.

```bash
python main.py prepare
python main.py train --test
python main.py train --test --wandb
python main.py train
python main.py eval
python main.py eval --model-path Qwen/Qwen2.5-0.5B-Instruct --max-samples 20
python main.py eval --wandb
python main.py --help
```

## Repository Layout

```text
.
├── data/                  # Generated train/val/test JSONL files
├── notebooks/             # Exploratory data analysis
├── scripts/
│   ├── prepare_data.py    # Dataset download + preprocessing
│   ├── train.py           # LoRA fine-tuning script
│   └── eval.py            # Adapter / model evaluation script
├── main.py                # CLI entry point
├── pyproject.toml         # Dependencies and project metadata
└── README.md
```

## Key Files

- [`main.py`](/Users/godblessjames/dev/medqa-finetune/main.py)
- [`scripts/prepare_data.py`](/Users/godblessjames/dev/medqa-finetune/scripts/prepare_data.py)
- [`scripts/train.py`](/Users/godblessjames/dev/medqa-finetune/scripts/train.py)
- [`scripts/eval.py`](/Users/godblessjames/dev/medqa-finetune/scripts/eval.py)
- [`pyproject.toml`](/Users/godblessjames/dev/medqa-finetune/pyproject.toml)

## License

This project is licensed under the MIT License. See [LICENSE](/Users/godblessjames/dev/medqa-finetune/LICENSE).
