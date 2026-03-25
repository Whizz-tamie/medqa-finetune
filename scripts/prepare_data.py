import os
import json
from tqdm import tqdm
from datasets import load_dataset

# Load MedQA dataset
train_data = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
test_data= load_dataset("GBaker/MedQA-USMLE-4-options", split="test")

# Split train into train + validation using the built-in method
split_data = train_data.train_test_split(test_size=0.05, seed=42)
train_data = split_data["train"]
val_data = split_data["test"]

# Model Prompt Configuration
SYSTEM_PROMPT = "You are a medical expert answering USMLE-style questions."
OUTPUT_DIR = "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_prompt_and_completion(example):
    """
    Constructs a conversational prompt-completion pair.
    """
    options = example['options']
    
    user_content = f"""Question:
{example['question']}

Options:
A. {options['A']}
B. {options['B']}
C. {options['C']}
D. {options['D']}

Answer with the letter (A, B, C, or D) preceded by ####."""

    target_content = f"#### {example['answer_idx']}"

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    completion = [
        {"role": "assistant", "content": target_content},
    ]

    return {"prompt": prompt, "completion": completion}

# Process and Save as JSONL
def export_to_jsonl(dataset, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for example in tqdm(dataset, desc=f"Exporting {os.path.basename(filename)}"):
            formatted_example = build_prompt_and_completion(example)
            json_line = json.dumps(formatted_example, ensure_ascii=False)
            f.write(json_line + "\n")
    print(f"Saved {len(dataset)} examples to {filename}")

export_to_jsonl(train_data, os.path.join(OUTPUT_DIR, "train.jsonl"))
export_to_jsonl(val_data, os.path.join(OUTPUT_DIR, "val.jsonl"))
export_to_jsonl(test_data, os.path.join(OUTPUT_DIR, "test.jsonl"))

print(f"\nSuccess! Data splits created: Train({len(train_data)}), Val({len(val_data)}), Test({len(test_data)})")
