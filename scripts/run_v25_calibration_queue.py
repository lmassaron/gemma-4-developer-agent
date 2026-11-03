#!/home/lmassaron/code/venv_gemma4_12b/bin/python
"""
Calibration Queue Runner for v25 Variants:
1. v25_rank6_svd on core_12 suite
2. v25_rank8_attn_svd on core_12 suite

Runs sequentially on GPU 0 with per-task JSON logging and summary outputs.
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

QUEUE = [
    {
        "name": "v25_rank6_svd",
        "lora_dir": str(ROOT / "models" / "lora_v25_rank6_svd"),
        "bundle_dir": str(ROOT / "state_of_the_art" / "bundle_v25_rank6_svd"),
        "label": "v25_rank6_svd_core12",
        "log_file": LOGS_DIR / "benchmark_v25_rank6_svd_core12.log",
        "results_file": LOGS_DIR / "benchmark_v25_rank6_svd_core12_results.json",
    },
    {
        "name": "v25_rank8_attn_svd",
        "lora_dir": str(ROOT / "models" / "lora_v25_rank8_attn_svd"),
        "bundle_dir": str(ROOT / "state_of_the_art" / "bundle_v25_rank8_attn_svd"),
        "label": "v25_rank8_attn_svd_core12",
        "log_file": LOGS_DIR / "benchmark_v25_rank8_attn_svd_core12.log",
        "results_file": LOGS_DIR / "benchmark_v25_rank8_attn_svd_core12_results.json",
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
    print(f" [CALIBRATION QUEUE] STARTING: {name} on core_12 suite", flush=True)
    print(f" LoRA Dir   : {lora_dir}", flush=True)
    print(f" Bundle Dir : {bundle_dir}", flush=True)
    print(f" Output JSON: {results_file.name}", flush=True)
    print(f" Log File   : {log_file.name}", flush=True)
    print(f"{'=' * 75}\n", flush=True)

    cmd = [
        PYTHON,
        str(EVAL_SCRIPT),
        "--suite", "core_12",
        "--label", label,
        "--lora_dir", lora_dir,
        "--bundle_dir", bundle_dir,
        "--output_file", str(results_file),
    ]

    with open(log_file, "w", encoding="utf-8") as f_out:
        f_out.write(f"\n--- [Calibration Queue] Starting {name} on core_12 at {time.ctime()} ---\n")
        f_out.flush()
        process = subprocess.Popen(
            cmd,
            stdout=f_out,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        process.wait()

    print(f"\n[CALIBRATION QUEUE] Finished {name} with return code {process.returncode}", flush=True)


def main():
    print(f"Calibration Queue Initialized with {len(QUEUE)} runs.", flush=True)
    for item in QUEUE:
        run_item(item)
    print("\nALL CALIBRATION RUNS FINISHED SUCCESSFULLY!", flush=True)


if __name__ == "__main__":
    main()

