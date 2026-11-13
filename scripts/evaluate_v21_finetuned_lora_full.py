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
    parser = argparse.ArgumentParser(description="Evaluate Zero-Shot or LoRA bundle on the SWE benchmark.")
    parser.add_argument("--suite", choices=["full_129", "core_60", "core_12", "guard_17", "calibration_24", "calibration_48"], default="core_12",
                        help="Benchmark suite to run (default: core_12)")
    parser.add_argument("--tasks_file", default=str(DATA_DIR / "tasks.jsonl"))
    parser.add_argument("--base_model", default="/home/lmassaron/models/kaggle/gemma-4-31b-it-qat-w4a16-ct")
    parser.add_argument("--lora_dir", default="none", help="Path to LoRA dir, or 'none' for Zero-Shot")
    parser.add_argument("--bundle_dir", default=None,
                        help="Path to submission bundle directory containing prompts and configs")
    parser.add_argument("--submission_zip", default=None,
                        help="Path to submission .zip file (overrides bundle_dir and auto-extracts adapter if present)")
    parser.add_argument("--label", default="v32_zero_shot_ap_bssr", help="Benchmark label for results")
    parser.add_argument("--timeout_sec", type=int, default=360)
    parser.add_argument("--max_turns", type=int, default=24)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--no_resume", action="store_true", help="Do not resume from existing results file")
    args = parser.parse_args()

    all_tasks = load_tasks(Path(args.tasks_file))
    task_map = {t["instance_id"]: t for t in all_tasks}

    if args.suite == "full_129":
        tasks = all_tasks
        default_out = ROOT / "state_of_the_art" / "benchmarks_data" / f"benchmark_{args.label}_full129_results.json"
    elif args.suite == "core_60":
        suite_file = ROOT / "state_of_the_art" / "benchmarks_data" / "test_tasks_core_60.json"
        suite_data = json.loads(suite_file.read_text(encoding="utf-8"))
        selected_ids = suite_data.get("tasks", [])
        tasks = [task_map[tid] for tid in selected_ids if tid in task_map]
        default_out = ROOT / "state_of_the_art" / "benchmarks_data" / f"benchmark_{args.label}_core60_results.json"
    elif args.suite in ("core_12", "guard_17", "calibration_24"):
        selected_ids = BENCHMARK_SUITES[args.suite]
        tasks = [task_map[tid] for tid in selected_ids if tid in task_map]
        default_out = ROOT / "state_of_the_art" / "benchmarks_data" / f"benchmark_{args.label}_{args.suite}_results.json"
    elif args.suite == "calibration_48":
        suite_file = ROOT / "state_of_the_art" / "benchmarks_data" / "test_tasks_v22_calibration_48.json"
        suite_data = json.loads(suite_file.read_text(encoding="utf-8"))
        selected_ids = suite_data.get("combined_48") or (suite_data.get("resolved_guard_17", []) + suite_data.get("near_misses_31", []))
        tasks = [task_map[tid] for tid in selected_ids if tid in task_map]
        default_out = ROOT / "state_of_the_art" / "benchmarks_data" / f"benchmark_{args.label}_calibration48_results.json"

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
    if not remaining_tasks:
        print("All tasks in suite already completed!")
        return

    device = "cuda:0"
    print(f"\nLoading processor from {args.base_model}...")
    processor = AutoProcessor.from_pretrained(args.base_model)

    if args.submission_zip and Path(args.submission_zip).is_file():
        import zipfile, shutil
        zip_p = Path(args.submission_zip).resolve()
        print(f"Using submission zip: {zip_p}")
        with zipfile.ZipFile(zip_p, "r") as z:
            namelist = z.namelist()
            prompt_str = z.read("prompts/system.md").decode("utf-8")
            sampling_cfg = yaml.safe_load(z.read("configs/sampling.yaml").decode("utf-8"))
            if (args.lora_dir == "none" or not args.lora_dir) and "agent.yaml" in namelist:
                ag_raw = z.read("agent.yaml").decode("utf-8")
                for ln in ag_raw.splitlines():
                    if ln.strip().startswith("adapter:"):
                        ad_name = ln.split(":", 1)[1].strip()
                        ad_prefix = f"adapters/{ad_name}/"
                        if any(n.startswith(ad_prefix) for n in namelist):
                            ext_dir = Path(f"/tmp/gemma4_extracted_adapters/{zip_p.stem}")
                            shutil.rmtree(ext_dir, ignore_errors=True)
                            ext_dir.mkdir(parents=True, exist_ok=True)
                            for m in namelist:
                                if m.startswith(ad_prefix):
                                    z.extract(m, ext_dir)
                            args.lora_dir = str(ext_dir / "adapters" / ad_name)
                            print(f"Auto-extracted LoRA adapter '{ad_name}' to {args.lora_dir}")
                        break
    else:
        bundle_dir = Path(args.bundle_dir or (ROOT / "state_of_the_art" / "bundle_v21_plus"))
        print(f"Using bundle configuration from: {bundle_dir}")
        prompt_str = (bundle_dir / "prompts" / "system.md").read_text(encoding="utf-8")
        sampling_cfg = yaml.safe_load((bundle_dir / "configs" / "sampling.yaml").read_text(encoding="utf-8"))

    print(f"Loading QAT model from {args.base_model} onto {device}...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        device_map=device,
    )
    if args.lora_dir and args.lora_dir.lower() != "none" and Path(args.lora_dir).is_dir():
        print(f"Attaching Fine-Tuned LoRA from {args.lora_dir}...")
        model = PeftModel.from_pretrained(model, args.lora_dir)
        print(f"Fine-tuned LoRA successfully attached in {time.time() - t0:.2f}s!")
    else:
        print(f"Zero-Shot QAT model ready in {time.time() - t0:.2f}s (no LoRA attached)!")

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
            "label": args.label,
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
