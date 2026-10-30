#!/usr/bin/env python3
"""
Run direct comparison between:
  1. Base Gemma 4 31B (v20 Budget-Fit)
  2. Gemma 4 31B + Scott Weeden SWE Tool-Use LoRA (v20_lora_scottweeden)
On the clean 12-task benchmark suite (core_12) under 100% faithful ADK simulator.
"""
import os
import sys
import json
import time
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
import yaml

from scripts.run_local_benchmark_leaderboard import (
    AgentSimulator,
    BENCHMARK_SUITES,
)

DATA_DIR = ROOT / "data"
TEMP_BASE_DIR = Path("/tmp/gemma4_benchmark_workspaces")
TEMP_BASE_DIR.mkdir(parents=True, exist_ok=True)

def load_tasks():
    tasks_file = DATA_DIR / "tasks.jsonl"
    tasks = []
    with open(tasks_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks

def run_suite(agent, tasks, label):
    print(f"\n=======================================================")
    print(f" STARTING BENCHMARK: {label} ({len(tasks)} tasks)")
    print(f"=======================================================")
    results = []
    resolved = 0
    start_t = time.time()
    
    for idx, t in enumerate(tasks, 1):
        tid = t["instance_id"]
        repo = t.get("repo", "")
        print(f"\n[{idx}/{len(tasks)}] Task: {tid} ({repo})")
        res = agent.solve_task(t, DATA_DIR, TEMP_BASE_DIR, timeout_sec=240, max_turns=20)
        is_res = res.get("resolved", False)
        if is_res:
            resolved += 1
        dur = res.get("duration_sec", 0)
        turns = res.get("turns", 0)
        patch_sz = res.get("patch_size", 0)
        print(f"  -> Result: status={res.get('status')} | resolved={is_res} | dur={dur}s | turns={turns} | patch={patch_sz}B")
        results.append(res)
        
    tot_dur = round(time.time() - start_t, 2)
    rate = round((resolved / len(tasks)) * 100, 2) if tasks else 0
    print(f"\n>>> {label} Summary: {resolved}/{len(tasks)} resolved ({rate}%) in {tot_dur}s")
    return {
        "label": label,
        "resolved": resolved,
        "total": len(tasks),
        "pass_rate": rate,
        "total_duration": tot_dur,
        "tasks": results,
    }

def main():
    suite_name = "core_12"
    all_tasks = load_tasks()
    suite_ids = set(BENCHMARK_SUITES[suite_name])
    tasks = [t for t in all_tasks if t["instance_id"] in suite_ids]
    
    base_model_path = "/home/lmassaron/models/kaggle/gemma-4-31b-it-qat-w4a16-ct"
    if not (Path(base_model_path) / "model.safetensors").exists():
        base_model_path = "google/gemma-4-31B-it"
        
    print(f"Using base model path: {base_model_path}")
    device = "cuda:0"
    
    processor = AutoProcessor.from_pretrained(base_model_path)
    
    v20_dir = ROOT / "state_of_the_art" / "bundle_v20"
    prompt_v20 = (v20_dir / "prompts" / "system.md").read_text(encoding="utf-8")
    sampling_v20 = yaml.safe_load((v20_dir / "configs" / "sampling.yaml").read_text(encoding="utf-8"))

    # 1. First run: Base Model (v20 Budget-Fit)
    print("\n--- Loading Model for Run 1: Base Model (v20 Budget-Fit) ---")
    model_base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if "qat" not in str(base_model_path) else "auto",
        device_map=device,
    )
    agent_base = AgentSimulator(model_base, processor, prompt_v20, sampling_v20, device=device)
    res_base = run_suite(agent_base, tasks, "v20_base_budget_fit")
    
    del model_base, agent_base
    torch.cuda.empty_cache()
    
    # 2. Second run: LoRA Model (v20 + Scott Weeden LoRA)
    print("\n--- Loading Model for Run 2: LoRA Model (v20 + Scott Weeden LoRA) ---")
    model_lora = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if "qat" not in str(base_model_path) else "auto",
        device_map=device,
    )
    lora_dir = "/home/lmassaron/models/kaggle/lora_scottweeden"
    print(f"Attaching LoRA from {lora_dir}...")
    model_lora = PeftModel.from_pretrained(model_lora, lora_dir)
    agent_lora = AgentSimulator(model_lora, processor, prompt_v20, sampling_v20, device=device)
    res_lora = run_suite(agent_lora, tasks, "v20_lora_scottweeden")
    
    del model_lora, agent_lora
    torch.cuda.empty_cache()
    
    # Save combined comparison
    out_file = ROOT / "state_of_the_art" / "benchmarks_data" / "benchmark_v20_vs_lora_core12_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({"base": res_base, "lora": res_lora}, indent=2), encoding="utf-8")
    print(f"\nComparison results written to {out_file}")

if __name__ == "__main__":
    main()
