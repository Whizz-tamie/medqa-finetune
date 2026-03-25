import sys
import subprocess
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_script(path, extra_args=None):
    """Helper to run a sub-script with optional arguments."""
    full_path = os.path.join(BASE_DIR, path)

    if not os.path.exists(full_path):
        print(f"[ERROR] Script not found: {path}")
        sys.exit(1)

    cmd = [sys.executable, full_path]
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"[DEBUG] Running: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Script failed: {path} (exit code {e.returncode})")
        sys.exit(e.returncode)

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [prepare|train|eval] [--test]")
        sys.exit(1)

    # Handle help anywhere in args
    if any(arg in ["-h", "--help"] for arg in sys.argv):
        print("Usage: python main.py [prepare|train|eval] [--test]")
        print("  prepare: Download and preprocess MedQA data")
        print("  train:   Start LoRA fine-tuning (Add --test for a smoke run)")
        print("  eval:    Run inference and check accuracy")
        sys.exit(0)

    command = sys.argv[1].lower()
    extra_args = sys.argv[2:]

    if command == "prepare":
        print("[INFO] Starting Data Preparation...")
        run_script("scripts/prepare_data.py")
    
    elif command == "train":
        if "--test" in extra_args:
            print("[INFO] Starting Training Pipeline (SMOKE TEST MODE)...")
        else:
            print("[INFO] Starting Training Pipeline (FULL RUN)...")
        run_script("scripts/train.py", extra_args)
        
    elif command == "eval":
        print("[INFO] Starting Evaluation...")
        run_script("scripts/eval.py", extra_args)

    else:
        print(f"[WARN] Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()