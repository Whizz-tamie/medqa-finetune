import sys
import subprocess
import os

def run_script(path):
    """Helper to run a sub-script and handle errors."""
    if not os.path.exists(path):
        print(f"[ERROR] Script not found: {path}")
        sys.exit(1)

    try:
        subprocess.run([sys.executable, path], check=True)
    except subprocess.CalledProcessError:
        print(f"[ERROR] Script failed: {path}")
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [prepare|train|eval]")
        print("  prepare: Download and preprocess MedQA data")
        print("  train:   Start LoRA fine-tuning (RTX 5090)")
        print("  eval:    Run inference and check accuracy")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command in ["-h", "--help"]:
        print("Usage: python main.py [prepare|train|eval]")
        sys.exit(0)

    if command == "prepare":
        print("[INFO] Starting Data Preparation...")
        run_script("scripts/prepare_data.py")
    
    elif command == "train":
        print("[INFO] Starting Training Pipeline...")
        run_script("scripts/train.py")
        
    elif command == "eval":
        print("[INFO] Starting Evaluation...")
        run_script("scripts/eval.py")

    else:
        print(f"[WARN] Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()