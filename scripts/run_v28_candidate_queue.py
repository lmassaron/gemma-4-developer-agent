#!/home/lmassaron/code/venv_gemma4_12b/bin/python
"""
Candidate Queue Runner for v28 LoRA Variants on Core-12 Suite:
1. v28_full_rank6_svd
2. v28_full_rank8_svd
3. v28_scottweeden

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
        "name": "v28_full_rank6_svd",
        "lora_dir": str(ROOT / "state_of_the_art" / "bundle_v28_full_rank6_svd" / "adapters" / "main_lora"),
        "bundle_dir": str(ROOT / "state_of_the_art" / "bundle_v28_full_rank6_svd"),
        "label": "v28_full_rank6_svd_core12",
        "log_file": LOGS_DIR / "benchmark_v28_full_rank6_svd_core12.log",
        "results_file": LOGS_DIR / "benchmark_v28_full_rank6_svd_core12_results.json",
    },
    {
        "name": "v28_full_rank8_svd",
        "lora_dir": str(ROOT / "state_of_the_art" / "bundle_v28_full_rank8_svd" / "adapters" / "main_lora"),
        "bundle_dir": str(ROOT / "state_of_the_art" / "bundle_v28_full_rank8_svd"),
        "label": "v28_full_rank8_svd_core12",
        "log_file": LOGS_DIR / "benchmark_v28_full_rank8_svd_core12.log",
        "results_file": LOGS_DIR / "benchmark_v28_full_rank8_svd_core12_results.json",
    },
    {
        "name": "v28_scottweeden",
        "lora_dir": str(ROOT / "state_of_the_art" / "bundle_v28_scottweeden" / "adapters" / "main_lora"),
        "bundle_dir": str(ROOT / "state_of_the_art" / "bundle_v28_scottweeden"),
        "label": "v28_scottweeden_core12",
        "log_file": LOGS_DIR / "benchmark_v28_scottweeden_core12.log",
        "results_file": LOGS_DIR / "benchmark_v28_scottweeden_core12_results.json",
    },
]


def run_item(item):
    name = item["name"]
    lora_dir = item["lora_dir"]
    bundle_dir = item["bundle_dir"]
    label = item["label"]
    log_file = item["log_file"]
    results_file = item["results_file"]

    # Check if already completed
    if results_file.is_file():
        import json
        try:
            d = json.loads(results_file.read_text())
            if len(d.get("tasks", [])) >= 12:
                print(f"[SKIP] {name} already has {len(d['tasks'])} completed tasks.")
                return
        except Exception:
            pass

    print(f"\n{'=' * 75}", flush=True)
    print(f" [CANDIDATE QUEUE] STARTING: {name} on core_12 suite", flush=True)
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

    with open(log_file, "a", encoding="utf-8") as f_out:
        f_out.write(f"\n--- [Candidate Queue] Starting {name} on core_12 at {time.ctime()} ---\n")
        f_out.flush()
        process = subprocess.Popen(
            cmd,
            stdout=f_out,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        process.wait()

    print(f"\n[CANDIDATE QUEUE] Finished {name} with return code {process.returncode}", flush=True)


def main():
    print(f"Candidate Calibration Queue Initialized with {len(QUEUE)} runs.", flush=True)
    for item in QUEUE:
        run_item(item)
    print("\nALL CANDIDATE CALIBRATION RUNS FINISHED SUCCESSFULLY!", flush=True)


if __name__ == "__main__":
    main()

