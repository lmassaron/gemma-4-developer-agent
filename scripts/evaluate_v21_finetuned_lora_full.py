#!/home/lmassaron/code/venv_gemma4_12b/bin/python
"""
Evaluate fine-tuned Scott Weeden LoRA (v21) on the complete SWE-bench benchmark.

Supports:
- full_129: All 129 SWE-bench tasks from data/tasks.jsonl
- core_60:  The standard 60-task core benchmark suite
- core_12:  The 12-task smoke/core suite

Includes incremental JSON output saving and automatic resumption from interruptions.
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import peft.tuners.lora.layer as lora_layer_module
import peft.tuners.lora.model as lora_model_module
import peft.tuners.tuners_utils as tu
from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear

# Monkey-patches to allow PEFT LoRA on compressed-tensors + Gemma4ClippableLinear
orig_tu_in_out = tu._get_in_out_features
def custom_tu_in_out(module):
    if hasattr(module, "in_features") and hasattr(module, "out_features"):
        return module.in_features, module.out_features
    if hasattr(module, "linear") and hasattr(module.linear, "in_features"):
        return module.linear.in_features, module.linear.out_features
    return orig_tu_in_out(module)
tu._get_in_out_features = custom_tu_in_out

def custom_layer_in_out(self, root_module):
    if hasattr(root_module, "in_features") and hasattr(root_module, "out_features"):
        return root_module.in_features, root_module.out_features
    if hasattr(root_module, "linear") and hasattr(root_module.linear, "in_features"):
        return root_module.linear.in_features, root_module.linear.out_features
    return orig_tu_in_out(root_module)
lora_layer_module.Linear._get_in_out_features = custom_layer_in_out

orig_create_new_module = lora_model_module.LoraModel._create_new_module
@staticmethod
def custom_create_new_module(lora_config, adapter_name, target, **kwargs):
    if isinstance(target, Gemma4ClippableLinear):
        return orig_create_new_module(lora_config, adapter_name, target.linear, **kwargs)
    return orig_create_new_module(lora_config, adapter_name, target, **kwargs)
lora_model_module.LoraModel._create_new_module = custom_create_new_module

from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
import yaml

from scripts.run_local_benchmark_leaderboard import (
    AgentSimulator,
    BENCHMARK_SUITES,
)

DATA_DIR = ROOT / "data"
TEMP_BASE_DIR = Path("/tmp/gemma4_benchmark_workspaces_full")
TEMP_BASE_DIR.mkdir(parents=True, exist_ok=True)


def load_tasks(tasks_file: Path):
    tasks = []
    with open(tasks_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Evaluate v21 LoRA on the complete SWE benchmark.")
    parser.add_argument("--suite", choices=["full_129", "core_60", "core_12"], default="full_129",
                        help="Benchmark suite to run (default: full_129)")
    parser.add_argument("--tasks_file", default=str(DATA_DIR / "tasks.jsonl"))
    parser.add_argument("--base_model", default="/home/lmassaron/models/kaggle/gemma-4-31b-it-qat-w4a16-ct")
    parser.add_argument("--lora_dir", default=str(ROOT / "models" / "lora_v21_scottweeden_finetuned"))
    parser.add_argument("--timeout_sec", type=int, default=240)
    parser.add_argument("--max_turns", type=int, default=20)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--no_resume", action="store_true", help="Do not resume from existing results file")
    args = parser.parse_args()

    all_tasks = load_tasks(Path(args.tasks_file))
    task_map = {t["instance_id"]: t for t in all_tasks}

    if args.suite == "full_129":
        tasks = all_tasks
        default_out = ROOT / "state_of_the_art" / "benchmarks_data" / "benchmark_v21_finetuned_lora_full129_results.json"
    elif args.suite == "core_60":
        suite_file = ROOT / "state_of_the_art" / "benchmarks_data" / "test_tasks_core_60.json"
        suite_data = json.loads(suite_file.read_text(encoding="utf-8"))
        selected_ids = suite_data.get("tasks", [])
        tasks = [task_map[tid] for tid in selected_ids if tid in task_map]
        default_out = ROOT / "state_of_the_art" / "benchmarks_data" / "benchmark_v21_finetuned_lora_core60_results.json"
    elif args.suite == "core_12":
        suite_ids = set(BENCHMARK_SUITES["core_12"])
        tasks = [t for t in all_tasks if t["instance_id"] in suite_ids]
        default_out = ROOT / "state_of_the_art" / "benchmarks_data" / "benchmark_v21_finetuned_lora_core12_results.json"

    out_file = Path(args.output_file) if args.output_file else default_out
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Check for resumption
    results = []
    completed_ids = set()
    if not args.no_resume and out_file.is_file():
        try:
            prev = json.loads(out_file.read_text(encoding="utf-8"))
            results = prev.get("tasks", [])
            completed_ids = {r["instance_id"] for r in results}
            print(f"[Resumption] Found {len(completed_ids)} already completed tasks in {out_file.name}.")
        except Exception as e:
            print(f"[Resumption warning] Could not read existing {out_file}: {e}")

    remaining_tasks = [t for t in tasks if t["instance_id"] not in completed_ids]
    print(f"Total benchmark tasks: {len(tasks)} | Completed: {len(completed_ids)} | Remaining: {len(remaining_tasks)}")

    device = "cuda:0"
    print(f"\nLoading processor from {args.base_model}...")
    processor = AutoProcessor.from_pretrained(args.base_model)

    bundle_dir = ROOT / "state_of_the_art" / "bundle_v21"
    prompt_str = (bundle_dir / "prompts" / "system.md").read_text(encoding="utf-8")
    sampling_cfg = yaml.safe_load((bundle_dir / "configs" / "sampling.yaml").read_text(encoding="utf-8"))

    print(f"Loading QAT model from {args.base_model} onto {device}...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        device_map=device,
    )
    print(f"Attaching Fine-Tuned LoRA from {args.lora_dir}...")
    model = PeftModel.from_pretrained(model, args.lora_dir)
    print(f"Fine-tuned LoRA successfully attached in {time.time() - t0:.2f}s!")

    agent = AgentSimulator(model, processor, prompt_str, sampling_cfg, device=device)

    print(f"\n=======================================================")
    print(f" STARTING BENCHMARK: v21_finetuned_lora | Suite: {args.suite} ({len(tasks)} tasks total)")
    print(f" Timeout: {args.timeout_sec}s | Max Turns: {args.max_turns} | Output: {out_file.name}")
    print(f"=======================================================")

    start_suite_time = time.time()

    for idx, t in enumerate(tasks, 1):
        tid = t["instance_id"]
        repo = t.get("repo", "")
        if tid in completed_ids:
            continue

        print(f"\n[{idx}/{len(tasks)}] Task: {tid} ({repo})", flush=True)
        task_start = time.time()
        res = agent.solve_task(t, DATA_DIR, TEMP_BASE_DIR, timeout_sec=args.timeout_sec, max_turns=args.max_turns)
        dur = res.get("duration_sec", round(time.time() - task_start, 2))
        is_res = res.get("resolved", False)
        status = res.get("status")
        turns = res.get("turns", 0)
        patch_sz = res.get("patch_size", 0)

        results.append(res)
        completed_ids.add(tid)

        current_resolved = sum(1 for r in results if r.get("resolved"))
        current_rate = round((current_resolved / len(results)) * 100, 2)
        print(f"  -> Result: status={status} | resolved={is_res} | dur={dur}s | turns={turns} | patch={patch_sz}B", flush=True)
        print(f"  -> [Running Score: {current_resolved}/{len(results)} ({current_rate}% resolved)]", flush=True)

        # Save checkpoint after each task
        checkpoint_data = {
            "label": "v21_finetuned_lora",
            "suite": args.suite,
            "resolved": current_resolved,
            "total_evaluated": len(results),
            "suite_total": len(tasks),
            "pass_rate": current_rate,
            "elapsed_time_sec": round(time.time() - start_suite_time, 2),
            "tasks": results,
        }
        out_file.write_text(json.dumps(checkpoint_data, indent=2), encoding="utf-8")

        # Memory hygiene
        gc.collect()
        torch.cuda.empty_cache()

    final_resolved = sum(1 for r in results if r.get("resolved"))
    final_rate = round((final_resolved / len(tasks)) * 100, 2) if tasks else 0.0
    total_time = round(time.time() - start_suite_time, 2)

    print(f"\n=======================================================")
    print(f" BENCHMARK COMPLETE: {final_resolved}/{len(tasks)} resolved ({final_rate}%) in {total_time}s")
    print(f" Results successfully saved to: {out_file.resolve()}")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
