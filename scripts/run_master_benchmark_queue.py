#!/home/lmassaron/code/venv_gemma4_12b/bin/python
"""
Master benchmark queue runner for full 129 SWE-bench suite:
Priority 1: v24_rank4_svd (Our new SVD compressed LoRA adapter, r=4, 127 MB)
Priority 2: v20_scottweeden (Resumes from task 26 to complete all 129)
Priority 3: v20_loopbreaker (Evaluates all 129 tasks)

All runs feature atomic per-task JSON saving, auto-resumption, and consecutive GPU scheduling.
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
BUNDLE_V22 = ROOT / "state_of_the_art" / "bundle_v22"
BUNDLE_V20 = ROOT / "state_of_the_art" / "bundle_v20"

QUEUE = [
    {
        "name": "v24_rank4_svd",
        "lora_dir": str(ROOT / "models" / "lora_v24_rank4_svd"),
        "bundle_dir": str(BUNDLE_V22),
        "label": "v24_rank4_svd_full129",
        "log_file": LOGS_DIR / "benchmark_v24_rank4_svd_full129.log",
        "results_file": LOGS_DIR / "benchmark_v24_rank4_svd_full129_results.json",
    },
    {
        "name": "v20_scottweeden",
        "lora_dir": "/home/lmassaron/models/kaggle/lora_scottweeden",
        "bundle_dir": str(BUNDLE_V20),
        "label": "v20_scottweeden_full129",
        "log_file": LOGS_DIR / "benchmark_v20_scottweeden_full129.log",
        "results_file": LOGS_DIR / "benchmark_v20_scottweeden_full129_results.json",
    },
    {
        "name": "v20_loopbreaker",
        "lora_dir": "/home/lmassaron/models/kaggle/lora_loopbreaker",
        "bundle_dir": str(BUNDLE_V20),
        "label": "v20_loopbreaker_full129",
        "log_file": LOGS_DIR / "benchmark_v20_loopbreaker_full129.log",
        "results_file": LOGS_DIR / "benchmark_v20_loopbreaker_full129_results.json",
    },
]


def run_item(item):
    name = item["name"]
    lora_dir = item["lora_dir"]
    bundle_dir = item["bundle_dir"]
    label = item["label"]
    log_file = item["log_file"]
    results_file = item["results_file"]

    print(f"\n{'=' * 75}", flush=True)
    print(f" [MASTER QUEUE] STARTING: {name} on full_129 suite", flush=True)
    print(f" LoRA Dir   : {lora_dir}", flush=True)
    print(f" Bundle Dir : {bundle_dir}", flush=True)
    print(f" Output JSON: {results_file.name}", flush=True)
    print(f" Log File   : {log_file.name}", flush=True)
    print(f"{'=' * 75}\n", flush=True)

    cmd = [
        PYTHON,
        str(EVAL_SCRIPT),
        "--suite", "full_129",
        "--label", label,
        "--lora_dir", lora_dir,
        "--bundle_dir", bundle_dir,
        "--output_file", str(results_file),
    ]

    with open(log_file, "a", encoding="utf-8") as f_out:
        f_out.write(f"\n--- [Master Queue] Starting {name} on full_129 at {time.ctime()} ---\n")
        f_out.flush()
        process = subprocess.Popen(
            cmd,
            stdout=f_out,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        process.wait()

    print(f"\n[MASTER QUEUE] Finished {name} with return code {process.returncode}", flush=True)


def main():
    print(f"Master Benchmark Queue Initialized with {len(QUEUE)} runs.", flush=True)
    for item in QUEUE:
        run_item(item)
    print("\nALL QUEUED RUNS FINISHED SUCCESSFULLY!", flush=True)


if __name__ == "__main__":
    main()

