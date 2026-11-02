#!/home/lmassaron/code/venv_gemma4_12b/bin/python
"""
Queue runner to benchmark community LoRAs on the full 129 SWE-bench dataset:
1. Scott Weeden LoRA (v20_scottweeden) on full_129
2. Loop-Breaker LoRA (v20_loopbreaker) on full_129

Includes incremental saving, resumption, and automatic consecutive execution.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/lmassaron/code/venv_gemma4_12b/bin/python"
EVAL_SCRIPT = ROOT / "scripts" / "evaluate_v21_finetuned_lora_full.py"
LOGS_DIR = ROOT / "state_of_the_art" / "benchmarks_data"
BUNDLE_V20 = ROOT / "state_of_the_art" / "bundle_v20"

RUNS = [
    {
        "name": "v20_scottweeden",
        "lora_dir": "/home/lmassaron/models/kaggle/lora_scottweeden",
        "label": "v20_scottweeden_full129",
        "log_file": LOGS_DIR / "benchmark_v20_scottweeden_full129.log",
        "results_file": LOGS_DIR / "benchmark_v20_scottweeden_full129_results.json",
    },
    {
        "name": "v20_loopbreaker",
        "lora_dir": "/home/lmassaron/models/kaggle/lora_loopbreaker",
        "label": "v20_loopbreaker_full129",
        "log_file": LOGS_DIR / "benchmark_v20_loopbreaker_full129.log",
        "results_file": LOGS_DIR / "benchmark_v20_loopbreaker_full129_results.json",
    },
]


def wait_for_idle_gpu():
    print("Checking GPU availability...")
    while True:
        # Check if evaluate_v21_finetuned_lora_full is running for core_12
        cmd = ["pgrep", "-f", "evaluate_v21_finetuned_lora_full.py --suite core_12"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if not res.stdout.strip():
            print("GPU is clear. Ready to begin queue.")
            break
        print("Waiting for core_12 evaluation to finish... (sleeping 30s)")
        time.sleep(30)


def run_benchmark(run_cfg):
    name = run_cfg["name"]
    lora_dir = run_cfg["lora_dir"]
    label = run_cfg["label"]
    log_file = run_cfg["log_file"]
    results_file = run_cfg["results_file"]

    print(f"\n{'=' * 75}")
    print(f" QUEUE LAUNCH: {name} on full_129 suite")
    print(f" LoRA Dir: {lora_dir}")
    print(f" Log File: {log_file}")
    print(f" Results : {results_file}")
    print(f"{'=' * 75}\n")

    cmd = [
        PYTHON,
        str(EVAL_SCRIPT),
        "--suite", "full_129",
        "--label", label,
        "--lora_dir", str(lora_dir),
        "--bundle_dir", str(BUNDLE_V20),
        "--output_file", str(results_file),
    ]

    with open(log_file, "a", encoding="utf-8") as f_out:
        f_out.write(f"\n--- Starting {name} on full_129 at {time.ctime()} ---\n")
        f_out.flush()
        process = subprocess.Popen(
            cmd,
            stdout=f_out,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        process.wait()

    print(f"\nFinished {name} with return code {process.returncode}")


def main():
    wait_for_idle_gpu()
    for run in RUNS:
        run_benchmark(run)
    print("\nALL QUEUED FULL 129 BENCHMARKS COMPLETE!")


if __name__ == "__main__":
    main()

