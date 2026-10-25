#!/usr/bin/env python3
"""
Side-by-side benchmark of v10 (lean thinking & decisive) vs v11 (targeted playbooks & inline edit) on DGX Spark.
Loads Gemma 4 31B once on GB10 GPU, runs both across standardized smoke suite tasks,
and outputs a comprehensive comparative table.
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
    ("v10_lean_thinking", "submissions_archive/submission_v10_lean_thinking_decisive.zip"),
    ("v11_targeted_playbooks", "submissions_archive/submission_v11_targeted_playbooks_inline_edit.zip"),
]

def load_submission(zip_path: Path):
    with zipfile.ZipFile(zip_path) as z:
        prompt = z.read("prompts/system.md").decode("utf-8")
        max_turns = 20
        timeout_sec = 330
        if "eval_config.yaml" in z.namelist():
            ev = yaml.safe_load(z.read("eval_config.yaml").decode("utf-8")) or {}
            cfg = ev.get("evaluation", {})
            max_turns = int(cfg.get("max_tool_calls", max_turns))
            timeout_sec = int(float(cfg.get("max_time_minutes", 5.5)) * 60)
        return prompt, max_turns, timeout_sec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="smoke_3", choices=["smoke_3", "core_12"])
    parser.add_argument("--base_model", default="google/gemma-4-31B-it")
    parser.add_argument("--tasks_file", default="data/tasks.jsonl")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--timeout_override", type=int, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_dir = root / args.data_dir
    temp_dir = Path("/tmp/benchmark_v10_v11_workspaces")
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Tasks
    target_ids = BENCHMARK_SUITES[args.suite]
    task_map = {}
    with open(root / args.tasks_file, "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            task_map[t["instance_id"]] = t
    tasks = [task_map[tid] for tid in target_ids if tid in task_map]
    print(f"Loaded {len(tasks)} benchmark tasks from suite '{args.suite}': {[t['instance_id'] for t in tasks]}")

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
        if args.timeout_override:
            timeout_sec = args.timeout_override
        print("=" * 70)
        print(f" BENCHMARKING: {name} ({zip_p.name})")
        print(f" Config: max_turns={max_turns}, timeout_sec={timeout_sec}s")
        print("=" * 70)

        agent = AgentSimulator(model, processor, prompt, device=device)
        cand_results = []
        for idx, t in enumerate(tasks, 1):
            print(f"\n--- [{idx}/{len(tasks)}] Candidate: {name} | Task: {t['instance_id']} ({t['repo']}) ---", flush=True)
            res = agent.solve_task(t, data_dir, temp_dir, timeout_sec=timeout_sec, max_turns=max_turns)
            cand_results.append(res)
            print(f"Result -> Status: {res['status']} | Resolved: {res['resolved']} | Duration: {res['duration_sec']}s | Turns: {res['turns']} | Patch: {res['patch_size']} bytes", flush=True)

        all_results[name] = cand_results

    # 4. Summary Matrix
    print("\n" + "=" * 80)
    print("                 HEAD-TO-HEAD BENCHMARK: v10 vs v11")
    print("=" * 80)
    print(f"{'Candidate':<24} | {'Resolved':<10} | {'Resolution %':<14} | {'Avg Time (s)':<14} | {'Avg Turns':<10}")
    print("-" * 80)
    for name, results in all_results.items():
        resolved = sum(1 for r in results if r["resolved"])
        total = len(results)
        rate = (resolved / total * 100.0) if total else 0.0
        avg_time = sum(r["duration_sec"] for r in results) / total if total else 0.0
        avg_turns = sum(r["turns"] for r in results) / total if total else 0.0
        print(f"{name:<24} | {f'{resolved}/{total}':<10} | {f'{rate:.1f}%':<14} | {f'{avg_time:.1f}s':<14} | {f'{avg_turns:.1f}':<10}")

    print("\n" + "=" * 80)
    print("                 PER-TASK DETAIL BREAKDOWN")
    print("=" * 80)
    print(f"{'Task':<16} | {'Metric':<14} | {'v10_lean_thinking':<24} | {'v11_targeted_playbooks':<24}")
    print("-" * 80)
    v10_res = {r['instance_id']: r for r in all_results.get('v10_lean_thinking', [])}
    v11_res = {r['instance_id']: r for r in all_results.get('v11_targeted_playbooks', [])}
    for t in tasks:
        tid = t['instance_id']
        r10 = v10_res.get(tid, {})
        r11 = v11_res.get(tid, {})
        print(f"{tid:<16} | {'Status':<14} | {r10.get('status', 'N/A'):<24} | {r11.get('status', 'N/A'):<24}")
        print(f"{'':<16} | {'Duration (s)':<14} | {str(r10.get('duration_sec', 'N/A')) + 's':<24} | {str(r11.get('duration_sec', 'N/A')) + 's':<24}")
        print(f"{'':<16} | {'Turns':<14} | {str(r10.get('turns', 'N/A')):<24} | {str(r11.get('turns', 'N/A')):<24}")
        print(f"{'':<16} | {'Patch Size':<14} | {str(r10.get('patch_size', 0)) + ' bytes':<24} | {str(r11.get('patch_size', 0)) + ' bytes':<24}")
        print("-" * 80)

    # Save detailed JSON
    out_json = root / "benchmark_v10_vs_v11.json"
    out_json.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved detailed comparison to {out_json}")

if __name__ == "__main__":
    main()
