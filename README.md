# medqa-finetune

Fine-tune `Qwen/Qwen2.5-7B-Instruct` on MedQA USMLE-style multiple-choice questions using LoRA, TRL, and Hugging Face tooling.

## What This Repo Does

This project turns the MedQA dataset into conversational prompt-completion examples and trains a Qwen instruct model to answer with the correct option letter in the format `#### A`.

By default, full training targets `Qwen/Qwen2.5-7B-Instruct`. The smaller `Qwen/Qwen2.5-0.5B-Instruct` model is used only for local smoke tests.

The current workflow is:

1. Download and preprocess MedQA into JSONL files
2. Run a lightweight local smoke test
3. Launch full LoRA fine-tuning

## Features

- MedQA preprocessing from Hugging Face datasets
- Conversational prompt-completion training format
- LoRA fine-tuning with PEFT
- TRL `SFTTrainer` training loop
- Safer `--test` mode for local validation
- Device-aware runtime settings for CUDA, Apple MPS, and CPU

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

- swaps to `Qwen/Qwen2.5-0.5B-Instruct`
- uses a small subset of the data
- uses a reduced-memory config
- runs only 5 training steps
- does not save checkpoints or a final model

### Run full training

```bash
python main.py train
```

The LoRA adapter is written to:

```text
outputs/qwen-medqa-adapter
```

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
- Smoke test model: `Qwen/Qwen2.5-0.5B-Instruct`
- Trainer: TRL `SFTTrainer`
- Fine-tuning method: LoRA via PEFT
- Loss setup: completion-only loss
- Default max sequence length: `512`

Runtime behavior:

- CUDA uses `bfloat16` and fused AdamW
- Apple MPS and CPU use `float32` and standard AdamW

Local smoke tests are intended for pipeline validation, not for judging final model quality. For reliable training results, use a CUDA machine.

## CLI

The repo uses [`main.py`](/Users/godblessjames/dev/medqa-finetune/main.py) as a small command entry point.

```bash
python main.py prepare
python main.py train --test
python main.py train
python main.py --help
```

## Repository Layout

```text
.
├── data/                  # Generated train/val/test JSONL files
├── notebooks/             # Exploratory data analysis
├── scripts/
│   ├── prepare_data.py    # Dataset download + preprocessing
│   └── train.py           # LoRA fine-tuning script
├── main.py                # CLI entry point
├── pyproject.toml         # Dependencies and project metadata
└── README.md
```

## Limitations

- `main.py` includes an `eval` command, but `scripts/eval.py` is not implemented yet.
- Apple Silicon smoke tests may still produce unstable metrics even when the training loop completes successfully.
- This repo currently focuses on preparation and training rather than evaluation or deployment.

## Key Files

- [`main.py`](/Users/godblessjames/dev/medqa-finetune/main.py)
- [`scripts/prepare_data.py`](/Users/godblessjames/dev/medqa-finetune/scripts/prepare_data.py)
- [`scripts/train.py`](/Users/godblessjames/dev/medqa-finetune/scripts/train.py)
- [`pyproject.toml`](/Users/godblessjames/dev/medqa-finetune/pyproject.toml)

## License

This project is licensed under the MIT License. See [LICENSE](/Users/godblessjames/dev/medqa-finetune/LICENSE).
