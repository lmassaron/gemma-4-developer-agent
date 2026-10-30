#!/usr/bin/env python3
"""
Comprehensive Head-to-Head Benchmark Runner: flexonafft_v2 vs v14 on DGX Spark.
Evaluates both candidates across the full 60-task suite (core_60).
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

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
scripts_dir = str(Path(__file__).resolve().parent)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from transformers import AutoProcessor
try:
    from transformers import AutoModelForImageTextToText as AutoModelForMultimodalLM
except ImportError:
    from transformers import AutoModelForMultimodalLM

from run_local_benchmark_leaderboard import AgentSimulator

CANDIDATES = {
    "flexonafft_v2": "submissions_archive/submission_flexonafft_v2.zip",
    "v14_heredoc_synthesis": "submissions_archive/submission_v14_heredoc_triage_synthesis.zip",
    "v15_surgical_triage": "submissions_archive/submission_v15.zip",
    "v16_phased_edit_enforcement": "submissions_archive/submission_v16_phased_edit_enforcement.zip",
    "v17_hill_climbed": "submissions_archive/submission_v17_hill_climbed.zip",
    "v18_hill_climbed": "submissions_archive/submission_v18_hill_climbed.zip",
    "v19_hill_climbed": "submissions_archive/submission_v19_hill_climbed.zip",
}

def load_submission(zip_path: Path):
    with zipfile.ZipFile(zip_path) as z:
        prompt = z.read("prompts/system.md").decode("utf-8")
        max_turns = 28
        timeout_sec = 480
        sampling_cfg = {}
        if "eval_config.yaml" in z.namelist():
            ev = yaml.safe_load(z.read("eval_config.yaml").decode("utf-8")) or {}
            cfg = ev.get("evaluation") if isinstance(ev.get("evaluation"), dict) else ev
            max_turns = int(cfg.get("max_tool_calls") or cfg.get("max_turns") or max_turns)
            timeout_sec = int(float(cfg.get("max_time_minutes", 8.0)) * 60)
        if "configs/sampling.yaml" in z.namelist():
            sampling_cfg = yaml.safe_load(z.read("configs/sampling.yaml").decode("utf-8")) or {}
        return prompt, max_turns, timeout_sec, sampling_cfg

def select_tasks(tasks_file: Path, num_tasks: int = 60, suite_file: Path = None):
    if suite_file and suite_file.is_file():
        if suite_file.suffix == ".jsonl":
            tasks = []
            with open(suite_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        tasks.append(json.loads(line))
            return tasks[:num_tasks]
        elif suite_file.suffix == ".json":
            task_map = {}
            with open(tasks_file, "r", encoding="utf-8") as f:
                for line in f:
                    t = json.loads(line)
                    task_map[t["instance_id"]] = t
            try:
                suite_data = json.loads(suite_file.read_text(encoding="utf-8"))
                selected_ids = suite_data.get("tasks", [])[:num_tasks]
                chosen = [task_map[tid] for tid in selected_ids if tid in task_map]
                if chosen:
                    return chosen
            except Exception as e:
                print(f"[Warning] Failed to load suite file {suite_file}: {e}", flush=True)

    task_map = {}
    with open(tasks_file, "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            task_map[t["instance_id"]] = t
    return list(task_map.values())[:num_tasks]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="all", choices=list(CANDIDATES.keys()) + ["all"])
    parser.add_argument("--suite", default="core_60", choices=["smoke_3", "core_60", "external_hf_py312"])
    parser.add_argument("--num_tasks", type=int, default=60)
    parser.add_argument("--base_model", default="google/gemma-4-31B-it")
    parser.add_argument("--tasks_file", default="data/tasks.jsonl")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_file", default="benchmark_flexonafft_vs_v14_results.json")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output_file checkpoint")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_dir = root / args.data_dir
    tasks_path = root / args.tasks_file
    temp_base = Path("/tmp/benchmark_workspaces_flexonafft_v14")
    temp_base.mkdir(parents=True, exist_ok=True)

    # 1. Load tasks
    if args.suite == "smoke_3":
        suite_file = root / "state_of_the_art" / "benchmarks_data" / "test_tasks_smoke_3.jsonl"
        args.num_tasks = min(args.num_tasks, 3)
    elif args.suite == "external_hf_py312":
        suite_file = root / "data" / "external_hf_benchmark_py312.jsonl"
        args.num_tasks = min(args.num_tasks, 12)
    else:
        suite_file = root / "state_of_the_art" / "benchmarks_data" / "test_tasks_core_60.json"
    tasks = select_tasks(tasks_path, num_tasks=args.num_tasks, suite_file=suite_file)
    print(f"Loaded {len(tasks)} benchmark tasks for evaluation ({args.suite})", flush=True)

    candidates_to_run = (
        list(CANDIDATES.keys()) if args.candidate == "all" else [args.candidate]
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 2. Load Model on GPU
    print(f"\n[Hardware / Model Setup] Loading {args.base_model} onto DGX Spark GPU ({device})...", flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.base_model)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded successfully in {time.time() - t0:.2f}s.\n", flush=True)

    all_results = {}

    for cand_name in candidates_to_run:
        zip_path = root / CANDIDATES[cand_name]
        assert zip_path.is_file(), f"Package not found: {zip_path}"

        prompt, max_turns, timeout_sec, sampling_cfg = load_submission(zip_path)

        print("\n" + "=" * 90, flush=True)
        print(f" BENCHMARKING CANDIDATE: {cand_name} ({zip_path.name})", flush=True)
        print(f" Tasks: {len(tasks)} | Max Turns: {max_turns} | Timeout/task: {timeout_sec}s", flush=True)
        print("=" * 90, flush=True)

        agent = AgentSimulator(model, processor, prompt, sampling_config=sampling_cfg, device=device)
        cand_results = []
        resolved_count = 0
        completed_ids = set()
        out_path = root / "state_of_the_art" / "benchmarks_data" / args.output_file

        if args.resume and out_path.is_file():
            try:
                existing_data = json.loads(out_path.read_text(encoding="utf-8"))
                all_results.update(existing_data)
                if cand_name in existing_data:
                    cand_results = existing_data[cand_name].get("tasks", [])
                    resolved_count = sum(1 for r in cand_results if r.get("resolved"))
                    completed_ids = {r.get("instance_id") for r in cand_results if r.get("instance_id")}
                    print(
                        f"[Resume] Loaded {len(cand_results)} completed tasks ({resolved_count} resolved) from {out_path.name}",
                        flush=True,
                    )
            except Exception as e:
                print(f"[Resume Warning] Could not load checkpoint {out_path}: {e}", flush=True)

        for idx, t in enumerate(tasks, 1):
            tid = t["instance_id"]
            if tid in completed_ids:
                continue
            repo = t.get("repo", "")
            cur_prompt, cur_max_turns, cur_timeout, cur_sampling = load_submission(zip_path)
            agent.system_prompt = cur_prompt
            agent.sampling_config = cur_sampling
            print(f"\n--- [{idx}/{len(tasks)}] Candidate: {cand_name} | Task: {tid} ({repo}) ---", flush=True)
            res = agent.solve_task(t, data_dir, temp_base, timeout_sec=cur_timeout, max_turns=cur_max_turns)
            cand_results.append(res)
            if res["resolved"]:
                resolved_count += 1
            print(f"Result -> Status: {res['status']} | Resolved: {res['resolved']} | Duration: {res['duration_sec']}s | Turns: {res['turns']} | Patch: {res['patch_size']} bytes | Running Total: {resolved_count}/{idx}", flush=True)

            # Save intermediate results after each task
            pass_rate = resolved_count / idx if idx else 0.0
            all_results[cand_name] = {
                "resolved": resolved_count,
                "completed": idx,
                "total": len(tasks),
                "pass_rate": round(pass_rate, 4),
                "tasks": cand_results,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)

    print(f"\nBenchmark completed! Results saved to {out_path}", flush=True)

if __name__ == "__main__":
    main()
