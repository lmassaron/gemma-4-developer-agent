#!/usr/bin/env python3
"""
Run local comparison benchmark of v7, v8, and v8b on DGX Spark (NVIDIA GB10).
Loads Gemma 4 31B once into memory, then benchmarks each submission archive
across the standardized SWE smoke suite.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
import yaml
import torch
from transformers import AutoProcessor
try:
    from transformers import AutoModelForImageTextToText as AutoModelForMultimodalLM
except ImportError:
    from transformers import AutoModelForMultimodalLM

from run_local_benchmark_leaderboard import BENCHMARK_SUITES, AgentSimulator, TaskWorkspace

CANDIDATES = [
    ("v7_pro", "submissions_archive/submission_v7_gemma4_31b_pro.zip"),
    ("v8_call_paced", "submissions_archive/submission_v8_call_paced_31b_pro.zip"),
    ("v8b_calibrated", "submissions_archive/submission_v8b_call_paced_32calls_5m.zip"),
]

def load_submission(zip_path: Path):
    with zipfile.ZipFile(zip_path) as z:
        prompt = z.read("prompts/system.md").decode("utf-8")
        max_turns = 20
        timeout_sec = 360
        if "eval_config.yaml" in z.namelist():
            ev = yaml.safe_load(z.read("eval_config.yaml").decode("utf-8")) or {}
            cfg = ev.get("evaluation", {})
            max_turns = int(cfg.get("max_tool_calls", max_turns))
            timeout_sec = int(float(cfg.get("max_time_minutes", 6)) * 60)
        return prompt, max_turns, timeout_sec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="smoke_3", choices=["smoke_3", "core_12"])
    parser.add_argument("--base_model", default="google/gemma-4-31B-it")
    parser.add_argument("--tasks_file", default="data/tasks.jsonl")
    parser.add_argument("--data_dir", default="data")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_dir = root / args.data_dir
    temp_dir = Path("/tmp/benchmark_comparison_workspaces")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Tasks
    target_ids = BENCHMARK_SUITES[args.suite]
    task_map = {}
    with open(root / args.tasks_file, "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            task_map[t["instance_id"]] = t
    tasks = [task_map[tid] for tid in target_ids if tid in task_map]
    print(f"Loaded {len(tasks)} benchmark tasks from suite '{args.suite}'.")

    # 2. Load Model Once
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n[Model] Loading Processor for {args.base_model}...")
    processor = AutoProcessor.from_pretrained(args.base_model)

    print(f"[Model] Loading Base Model: {args.base_model} in bfloat16 on {device}...")
    t0 = time.time()
    model = AutoModelForMultimodalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded successfully in {time.time() - t0:.2f}s.\n")

    # 3. Benchmark Each Candidate
    all_results = {}
    for name, rel_zip in CANDIDATES:
        zip_p = root / rel_zip
        if not zip_p.is_file():
            print(f"Skipping missing package: {zip_p}")
            continue

        prompt, max_turns, timeout_sec = load_submission(zip_p)
        print("=" * 65)
        print(f" BENCHMARKING: {name} ({zip_p.name})")
        print(f" Config: max_turns={max_turns}, timeout_sec={timeout_sec}s")
        print("=" * 65)

        agent = AgentSimulator(model, processor, prompt, device=device)
        cand_results = []
        for idx, t in enumerate(tasks, 1):
            print(f"\n--- [{idx}/{len(tasks)}] Task: {t['instance_id']} ({t['repo']}) ---")
            res = agent.solve_task(t, data_dir, temp_dir, timeout_sec=timeout_sec, max_turns=max_turns)
            cand_results.append(res)
            print(f"Result: status={res['status']} | resolved={res['resolved']} | duration={res['duration_sec']}s | turns={res['turns']}")

        all_results[name] = cand_results

    # 4. Summary Table
    print("\n" + "=" * 65)
    print("                BENCHMARK COMPARISON SUMMARY")
    print("=" * 65)
    print(f"{'Candidate':<18} | {'Resolved':<10} | {'Resolution %':<14} | {'Avg Time (s)':<14} | {'Avg Turns':<10}")
    print("-" * 75)
    for name, results in all_results.items():
        resolved = sum(1 for r in results if r["resolved"])
        total = len(results)
        rate = (resolved / total * 100.0) if total else 0.0
        avg_time = sum(r["duration_sec"] for r in results) / total if total else 0.0
        avg_turns = sum(r["turns"] for r in results) / total if total else 0.0
        print(f"{name:<18} | {f'{resolved}/{total}':<10} | {f'{rate:.1f}%':<14} | {f'{avg_time:.1f}s':<14} | {f'{avg_turns:.1f}':<10}")

    # Write results JSON
    out_json = root / "benchmark_comparison_results.json"
    out_json.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved detailed results to {out_json}")

if __name__ == "__main__":
    main()
