#!/usr/bin/env python3
"""
Evaluate LoRA-augmented quantized Gemma 4 31B (Robin Amir Bahar - Loop Breaker) on core_12.
"""
import os
import sys
import json
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

def main():
    suite_name = "core_12"
    all_tasks = load_tasks()
    suite_ids = set(BENCHMARK_SUITES[suite_name])
    tasks = [t for t in all_tasks if t["instance_id"] in suite_ids]
    
    base_model_path = "/home/lmassaron/models/kaggle/gemma-4-31b-it-qat-w4a16-ct"
    lora_dir = "/home/lmassaron/models/kaggle/lora_robinamirbahar"
    device = "cuda:0"
    
    print(f"Loading processor from {base_model_path}...")
    processor = AutoProcessor.from_pretrained(base_model_path)
    
    v20_dir = ROOT / "state_of_the_art" / "bundle_v20"
    prompt_v20 = (v20_dir / "prompts" / "system.md").read_text(encoding="utf-8")
    sampling_v20 = yaml.safe_load((v20_dir / "configs" / "sampling.yaml").read_text(encoding="utf-8"))
    
    print(f"Loading QAT model from {base_model_path} onto {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        device_map=device,
    )
    print(f"Attaching LoRA from {lora_dir}...")
    model = PeftModel.from_pretrained(model, lora_dir)
    print("LoRA successfully attached to model!")
    
    agent = AgentSimulator(model, processor, prompt_v20, sampling_v20, device=device)
    
    print(f"\n=======================================================")
    print(f" STARTING BENCHMARK: v20_lora_loopbreaker ({len(tasks)} tasks)")
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
    print(f"\n>>> v20_lora_loopbreaker Summary: {resolved}/{len(tasks)} resolved ({rate}%) in {tot_dur}s")
    
    out_file = ROOT / "state_of_the_art" / "benchmarks_data" / "benchmark_v20_loopbreaker_core12_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({
        "label": "v20_lora_loopbreaker",
        "resolved": resolved,
        "total": len(tasks),
        "pass_rate": rate,
        "total_duration": tot_dur,
        "tasks": results
    }, indent=2), encoding="utf-8")
    print(f"\nResults written to {out_file}")

if __name__ == "__main__":
    main()

