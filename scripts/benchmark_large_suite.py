#!/usr/bin/env python3
"""
Comprehensive 60-Task Benchmark Runner on DGX Spark (Option A: Dynamic Git Worktree).
Evaluates candidate submissions across up to 60 diverse SWE-bench tasks from data/tasks.jsonl.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import shutil
import subprocess
import time
import zipfile
import yaml
from transformers import AutoProcessor
try:
    from transformers import AutoModelForImageTextToText as AutoModelForMultimodalLM
except ImportError:
    from transformers import AutoModelForMultimodalLM

from run_local_benchmark_leaderboard import BENCHMARK_SUITES, AgentSimulator, TaskWorkspace

CANDIDATES = {
    "v10_lean_thinking": "submissions_archive/submission_v10_lean_thinking_decisive.zip",
    "v11_targeted_playbooks": "submissions_archive/submission_v11_targeted_playbooks_inline_edit.zip",
    "v12_native_discipline": "submissions_archive/submission_v12_native_tool_discipline_atomic_verify.zip",
    "v13_non_empty_patch": "submissions_archive/submission_v13_non_empty_patch_enforced.zip",
    "v13_claude_clean": "submissions_archive/submission_v13_heredoc_diffcheck_clean.zip",
}

def load_submission(zip_path: Path):
    with zipfile.ZipFile(zip_path) as z:
        prompt = z.read("prompts/system.md").decode("utf-8")
        max_turns = 30
        timeout_sec = 300
        if "eval_config.yaml" in z.namelist():
            ev = yaml.safe_load(z.read("eval_config.yaml").decode("utf-8")) or {}
            cfg = ev.get("evaluation") if isinstance(ev.get("evaluation"), dict) else ev
            max_turns = int(cfg.get("max_tool_calls") or cfg.get("max_turns") or max_turns)
            timeout_sec = int(float(cfg.get("max_time_minutes", 5.0)) * 60)
        return prompt, max_turns, timeout_sec

def select_tasks(tasks_file: Path, num_tasks: int = 60, suite_file: Path = None):
    task_map = {}
    with open(tasks_file, "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            task_map[t["instance_id"]] = t

    if suite_file and suite_file.is_file():
        try:
            suite_data = json.loads(suite_file.read_text(encoding="utf-8"))
            selected_ids = suite_data.get("tasks", [])[:num_tasks]
            chosen = [task_map[tid] for tid in selected_ids if tid in task_map]
            if chosen:
                return chosen
        except Exception as e:
            print(f"[Warning] Failed to load suite file {suite_file}: {e}", flush=True)

    return list(task_map.values())[:num_tasks]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="v13_non_empty_patch", choices=list(CANDIDATES.keys()) + ["all"])
    parser.add_argument("--num_tasks", type=int, default=60, help="Number of benchmark tasks (default: 60)")
    parser.add_argument("--base_model", default="google/gemma-4-31B-it")
    parser.add_argument("--tasks_file", default="data/tasks.jsonl")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--timeout_override", type=int, default=None)
    parser.add_argument("--output_file", default="benchmark_core_60_results.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_dir = root / args.data_dir
    tasks_path = root / args.tasks_file
    temp_base = Path("/tmp/benchmark_workspaces_60")
    if temp_base.exists():
        shutil.rmtree(temp_base, ignore_errors=True)
    temp_base.mkdir(parents=True, exist_ok=True)

    # 1. Load Tasks
    suite_file = root / "state_of_the_art" / "benchmarks_data" / "test_tasks_core_60.json"
    tasks = select_tasks(tasks_path, num_tasks=args.num_tasks, suite_file=suite_file)
    print(f"Loaded {len(tasks)} benchmark tasks for evaluation:", flush=True)
    repos = {}
    for t in tasks:
        r = t.get("repo", "unknown")
        repos[r] = repos.get(r, 0) + 1
    for r, cnt in repos.items():
        print(f"  - {r}: {cnt} tasks", flush=True)

    # 2. Load Model
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

    # 3. Determine Candidates to Run
    targets = [args.candidate] if args.candidate != "all" else list(CANDIDATES.keys())
    all_results = {}

    for cand_name in targets:
        zip_rel = CANDIDATES[cand_name]
        zip_path = root / zip_rel
        assert zip_path.is_file(), f"Package not found: {zip_path}"

        prompt, max_turns, timeout_sec = load_submission(zip_path)
        if args.timeout_override:
            timeout_sec = args.timeout_override

        print("\n" + "=" * 90, flush=True)
        print(f" BENCHMARKING CANDIDATE: {cand_name} ({zip_path.name})", flush=True)
        print(f" Tasks: {len(tasks)} | Max Turns: {max_turns} | Timeout/task: {timeout_sec}s", flush=True)
        print("=" * 90, flush=True)

        agent = AgentSimulator(model, processor, prompt, device=device)
        cand_results = []
        resolved_count = 0

        for idx, t in enumerate(tasks, 1):
            tid = t["instance_id"]
            repo = t.get("repo", "")
            print(f"\n--- [{idx}/{len(tasks)}] Candidate: {cand_name} | Task: {tid} ({repo}) ---", flush=True)
            t_start = time.time()
            res = agent.solve_task(t, data_dir, temp_base, timeout_sec=timeout_sec, max_turns=max_turns)
            cand_results.append(res)
            if res["resolved"]:
                resolved_count += 1
            print(f"Result -> Status: {res['status']} | Resolved: {res['resolved']} | Duration: {res['duration_sec']}s | Turns: {res['turns']} | Patch: {res['patch_size']} bytes | Running Total: {resolved_count}/{idx}", flush=True)

        all_results[cand_name] = cand_results

    # 4. Save and Report Matrix
    out_path = root / args.output_file
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nDetailed evaluation results saved to: {out_path}", flush=True)

    print("\n" + "=" * 95, flush=True)
    print("                    60-TASK BENCHMARK EVALUATION MATRIX", flush=True)
    print("=" * 95, flush=True)
    print(f"{'Candidate':<26} | {'Resolved':<12} | {'Resolution %':<14} | {'Avg Time (s)':<14} | {'Avg Turns':<10}", flush=True)
    print("-" * 95, flush=True)
    for c_name, results in all_results.items():
        resolved = sum(1 for r in results if r["resolved"])
        total = len(results)
        rate = (resolved / total * 100.0) if total else 0.0
        avg_time = sum(r["duration_sec"] for r in results) / total if total else 0.0
        avg_turns = sum(r["turns"] for r in results) / total if total else 0.0
        print(f"{c_name:<26} | {f'{resolved}/{total}':<12} | {f'{rate:.1f}%':<14} | {f'{avg_time:.1f}s':<14} | {f'{avg_turns:.1f}':<10}", flush=True)
    print("=" * 95, flush=True)

if __name__ == "__main__":
    main()
