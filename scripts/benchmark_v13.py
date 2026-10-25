#!/usr/bin/env python3
"""
Benchmark v13 (Non-Empty Patch Enforced) on DGX Spark,
comparing directly against v10, v11, and v12 baselines.
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

TARGET_CANDIDATE = ("v13_non_empty_patch", "submissions_archive/submission_v13_non_empty_patch_enforced.zip")

def load_submission(zip_path: Path):
    with zipfile.ZipFile(zip_path) as z:
        prompt = z.read("prompts/system.md").decode("utf-8")
        max_turns = 30
        timeout_sec = 330
        if "eval_config.yaml" in z.namelist():
            ev = yaml.safe_load(z.read("eval_config.yaml").decode("utf-8")) or {}
            cfg = ev.get("evaluation") if isinstance(ev.get("evaluation"), dict) else ev
            max_turns = int(cfg.get("max_tool_calls") or cfg.get("max_turns") or max_turns)
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
    temp_dir = Path("/tmp/benchmark_v13_workspaces")
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
    print(f"Loaded {len(tasks)} benchmark tasks from suite '{args.suite}': {[t['instance_id'] for t in tasks]}", flush=True)

    # 2. Load Model Once
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n[Model] Loading Processor for {args.base_model}...", flush=True)
    processor = AutoProcessor.from_pretrained(args.base_model)

    print(f"[Model] Loading Base Model: {args.base_model} in bfloat16 on {device}...", flush=True)
    t0 = time.time()
    model = AutoModelForMultimodalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded successfully in {time.time() - t0:.2f}s.\n", flush=True)

    # 3. Load Cached Baseline Results (v10, v11, and v12)
    all_results = {}
    cached_file = root / "benchmark_v10_v11_v12.json"
    if cached_file.is_file():
        try:
            cached_data = json.loads(cached_file.read_text(encoding="utf-8"))
            for k in ["v10_lean_thinking", "v11_targeted_playbooks", "v12_native_discipline"]:
                if k in cached_data:
                    all_results[k] = cached_data[k]
                    print(f"Loaded cached baseline for {k} ({len(all_results[k])} tasks)", flush=True)
        except Exception as e:
            print(f"Notice: Failed to load cached v10/v11/v12 baselines: {e}", flush=True)

    # 4. Benchmark v13 Candidate
    name, rel_zip = TARGET_CANDIDATE
    zip_p = root / rel_zip
    assert zip_p.is_file(), f"Target submission package not found: {zip_p}"

    prompt, max_turns, timeout_sec = load_submission(zip_p)
    if args.timeout_override:
        timeout_sec = args.timeout_override
    print("=" * 90, flush=True)
    print(f" BENCHMARKING: {name} ({zip_p.name})", flush=True)
    print(f" Config: max_turns={max_turns}, timeout_sec={timeout_sec}s", flush=True)
    print("=" * 90, flush=True)

    agent = AgentSimulator(model, processor, prompt, device=device)
    v13_results = []
    for idx, t in enumerate(tasks, 1):
        print(f"\n--- [{idx}/{len(tasks)}] Candidate: {name} | Task: {t['instance_id']} ({t['repo']}) ---", flush=True)
        res = agent.solve_task(t, data_dir, temp_dir, timeout_sec=timeout_sec, max_turns=max_turns)
        v13_results.append(res)
        print(f"Result -> Status: {res['status']} | Resolved: {res['resolved']} | Duration: {res['duration_sec']}s | Turns: {res['turns']} | Patch: {res['patch_size']} bytes", flush=True)

    all_results[name] = v13_results

    # 5. Summary Matrix (v10 vs v11 vs v12 vs v13)
    print("\n" + "=" * 105, flush=True)
    print("                 HEAD-TO-HEAD BENCHMARK: v10 vs v11 vs v12 vs v13", flush=True)
    print("=" * 105, flush=True)
    print(f"{'Candidate':<26} | {'Resolved':<10} | {'Resolution %':<14} | {'Avg Time (s)':<14} | {'Avg Turns':<10}", flush=True)
    print("-" * 105, flush=True)
    for c_name, results in all_results.items():
        resolved = sum(1 for r in results if r["resolved"])
        total = len(results)
        rate = (resolved / total * 100.0) if total else 0.0
        avg_time = sum(r["duration_sec"] for r in results) / total if total else 0.0
        avg_turns = sum(r["turns"] for r in results) / total if total else 0.0
        print(f"{c_name:<26} | {f'{resolved}/{total}':<10} | {f'{rate:.1f}%':<14} | {f'{avg_time:.1f}s':<14} | {f'{avg_turns:.1f}':<10}", flush=True)

    print("\n" + "=" * 125, flush=True)
    print("                                              PER-TASK DETAIL BREAKDOWN", flush=True)
    print("=" * 125, flush=True)
    print(f"{'Task':<16} | {'Metric':<14} | {'v10_lean':<18} | {'v11_targeted':<18} | {'v12_discipline':<18} | {'v13_non_empty':<18}", flush=True)
    print("-" * 125, flush=True)
    v10_res = {r['instance_id']: r for r in all_results.get('v10_lean_thinking', [])}
    v11_res = {r['instance_id']: r for r in all_results.get('v11_targeted_playbooks', [])}
    v12_res = {r['instance_id']: r for r in all_results.get('v12_native_discipline', [])}
    v13_res = {r['instance_id']: r for r in all_results.get('v13_non_empty_patch', [])}
    for t in tasks:
        tid = t['instance_id']
        r10 = v10_res.get(tid, {})
        r11 = v11_res.get(tid, {})
        r12 = v12_res.get(tid, {})
        r13 = v13_res.get(tid, {})
        print(f"{tid:<16} | {'Status':<14} | {r10.get('status', 'N/A'):<18} | {r11.get('status', 'N/A'):<18} | {r12.get('status', 'N/A'):<18} | {r13.get('status', 'N/A'):<18}", flush=True)
        print(f"{'':<16} | {'Duration (s)':<14} | {str(r10.get('duration_sec', 'N/A')) + 's':<18} | {str(r11.get('duration_sec', 'N/A')) + 's':<18} | {str(r12.get('duration_sec', 'N/A')) + 's':<18} | {str(r13.get('duration_sec', 'N/A')) + 's':<18}", flush=True)
        print(f"{'':<16} | {'Turns':<14} | {str(r10.get('turns', 'N/A')):<18} | {str(r11.get('turns', 'N/A')):<18} | {str(r12.get('turns', 'N/A')):<18} | {str(r13.get('turns', 'N/A')):<18}", flush=True)
        print(f"{'':<16} | {'Patch Size':<14} | {str(r10.get('patch_size', 0)) + ' B':<18} | {str(r11.get('patch_size', 0)) + ' B':<18} | {str(r12.get('patch_size', 0)) + ' B':<18} | {str(r13.get('patch_size', 0)) + ' B':<18}", flush=True)
        print("-" * 125, flush=True)

    # Save detailed JSONs
    out_json = root / "benchmark_v10_v11_v12_v13.json"
    out_json.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved detailed comparison to {out_json}", flush=True)

    sota_json = root / "state_of_the_art" / "benchmarks_data" / "benchmark_v10_v11_v12_v13.json"
    sota_json.write_text(json.dumps(all_results, indent=2))

if __name__ == "__main__":
    main()
