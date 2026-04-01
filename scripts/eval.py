import argparse
import json
import os
import re
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb


DEFAULT_ADAPTER_PATH = "outputs/qwen-medqa-adapter"
DEFAULT_DATA_PATH = "data/test.jsonl"
DEFAULT_WANDB_PROJECT = "medqa-qwen-finetune"


def get_runtime_settings():
    if torch.cuda.is_available():
        return {
            "device_name": torch.cuda.get_device_name(0),
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "target_device": None,
        }

    return {
        "device_name": "CPU",
        "torch_dtype": torch.float32,
        "device_map": None,
        "target_device": "cpu",
    }


def extract_answer_letter(text):
    match = re.search(r"####\s*([ABCD])", text)
    if match:
        return match.group(1)

    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)

    return None


def load_examples(data_path, max_samples=None):
    examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
            if max_samples is not None and len(examples) >= max_samples:
                break
    return examples


def get_question_text(example):
    prompt_messages = example.get("prompt", [])
    for message in reversed(prompt_messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def extract_options(example):
    user_text = get_question_text(example)
    matches = re.findall(r"^([ABCD])\.\s*(.+)$", user_text, flags=re.MULTILINE)
    return {letter: text.strip() for letter, text in matches}


def resolve_model_paths(adapter_path_arg, model_path_arg):
    if model_path_arg:
        return None, model_path_arg

    adapter_path = Path(adapter_path_arg) if adapter_path_arg else None
    if adapter_path is not None and adapter_path.exists():
        peft_config = PeftConfig.from_pretrained(adapter_path)
        base_model_id = model_path_arg or peft_config.base_model_name_or_path
        return adapter_path, base_model_id

    raise FileNotFoundError(
        "No valid evaluation target found. Provide --adapter-path for a saved LoRA adapter "
        "or --model-path for a standalone model."
    )


def build_run_name(adapter_path, base_model_id):
    if adapter_path is not None:
        return f"{Path(adapter_path).name}-eval"
    return f"{base_model_id.split('/')[-1]}-eval"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH, help="Path to the saved LoRA adapter")
    parser.add_argument("--model-path", default=None, help="Optional standalone model path or model id")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Path to the evaluation JSONL file")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on the number of examples")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="Maximum number of tokens to generate")
    parser.add_argument("--show-samples", type=int, default=5, help="Number of sample predictions to print")
    parser.add_argument("--wandb-samples", type=int, default=200, help="Number of sample predictions to log to W&B")
    parser.add_argument("--wandb", action="store_true", help="Log evaluation metrics to Weights & Biases")
    parser.add_argument("--wandb-run-name", default=None, help="Optional override for the W&B run name")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Evaluation data not found: {data_path}")

    runtime_settings = get_runtime_settings()
    adapter_path, base_model_id = resolve_model_paths(args.adapter_path, args.model_path)
    run_name = args.wandb_run_name or build_run_name(adapter_path, base_model_id)

    print(f"[INFO] Loading base model: {base_model_id}")
    if adapter_path is not None:
        print(f"[INFO] Loading adapter from: {adapter_path}")
    else:
        print("[INFO] No adapter path provided. Evaluating the base/standalone model directly.")
    print(
        f"[INFO] Evaluation runtime: {runtime_settings['device_name']} "
        f"(dtype={runtime_settings['torch_dtype']})"
    )

    wandb_run = None
    if args.wandb:
        os.environ.setdefault("WANDB_PROJECT", DEFAULT_WANDB_PROJECT)
        wandb_run = wandb.init(
            project=os.environ["WANDB_PROJECT"],
            name=run_name,
            config={
                "adapter_path": str(adapter_path) if adapter_path is not None else None,
                "model_path": base_model_id,
                "data_path": str(data_path),
                "max_samples": args.max_samples,
                "max_new_tokens": args.max_new_tokens,
                "runtime_device": runtime_settings["device_name"],
                "runtime_dtype": str(runtime_settings["torch_dtype"]),
            },
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=runtime_settings["torch_dtype"],
        device_map=runtime_settings["device_map"],
        trust_remote_code=True,
    )
    if runtime_settings["target_device"] is not None:
        model = model.to(runtime_settings["target_device"])

    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    model_input_device = next(model.parameters()).device

    examples = load_examples(data_path, max_samples=args.max_samples)
    correct = 0
    invalid = 0
    predictions = []

    for example in tqdm(examples, desc="Evaluating"):
        prompt_messages = example["prompt"]
        target_messages = example["completion"]
        options = extract_options(example)
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(model_input_device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        predicted = extract_answer_letter(generated_text)
        expected = extract_answer_letter(target_messages[0]["content"])
        invalid += int(predicted is None)
        is_correct = predicted == expected
        correct += int(is_correct)

        predictions.append(
            {
                "question": get_question_text(example),
                "predicted": predicted,
                "predicted_option_text": options.get(predicted),
                "expected": expected,
                "expected_option_text": options.get(expected),
                "generated_text": generated_text,
                "is_correct": is_correct,
            }
        )

    total = len(predictions)
    accuracy = (correct / total) if total else 0.0
    print(f"\n[RESULT] Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"[RESULT] Invalid predictions: {invalid}")

    if wandb_run is not None:
        wandb.log(
            {
                "eval/accuracy": accuracy,
                "eval/correct": correct,
                "eval/num_examples": total,
                "eval/invalid_predictions": invalid,
            }
        )

    sample_count = min(args.show_samples, total)
    if sample_count:
        print("\n[RESULT] Sample predictions:")
        for idx, prediction in enumerate(predictions[:sample_count], start=1):
            print(
                f"{idx}. expected={prediction['expected']} "
                f"predicted={prediction['predicted']} "
                f"correct={prediction['is_correct']} "
                f"text={prediction['generated_text']!r}"
            )

    wandb_sample_count = min(args.wandb_samples, total)
    if wandb_run is not None and wandb_sample_count:
        sample_table = wandb.Table(
            columns=[
                "question",
                "expected",
                "expected_option_text",
                "predicted",
                "predicted_option_text",
                "correct",
                "generated_text",
            ]
        )
        for prediction in predictions[:wandb_sample_count]:
            sample_table.add_data(
                prediction["question"],
                prediction["expected"],
                prediction["expected_option_text"],
                prediction["predicted"],
                prediction["predicted_option_text"],
                prediction["is_correct"],
                prediction["generated_text"],
            )
        wandb.log({"eval/sample_predictions": sample_table})

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
