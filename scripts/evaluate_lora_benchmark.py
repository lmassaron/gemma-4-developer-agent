#!/usr/bin/env python3
"""
Evaluate LoRA vs Base Gemma 4 on the verified 12-task benchmark suite (core_12).
Uses the clean Kaggle-faithful ADK tool-calling simulator without heuristic overrides.
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_local_benchmark_leaderboard import (
    AgentSimulator,
    BENCHMARK_SUITES,
    load_tasks,
    DATA_DIR,
    TEMP_BASE_DIR,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="google/gemma-4-31B-it")
    parser.add_argument("--lora-path", default=None, help="Path to LoRA adapter directory")
    parser.add_argument("--bundle-dir", default="state_of_the_art/bundle_v20")
    parser.add_argument("--suite", default="smoke_3", choices=["smoke_3", "core_12"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    bundle_dir = ROOT / args.bundle_dir
    prompt_path = bundle_dir / "prompts" / "system.md"
    sampling_path = bundle_dir / "configs" / "sampling.yaml"
    import yaml
    system_prompt = prompt_path.read_text(encoding="utf-8")
    sampling_config = yaml.safe_load(sampling_path.read_text(encoding="utf-8")) if sampling_path.exists() else {}

    print(f"=== Loading Base Model: {args.model_path} ===")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    if args.lora_path:
        print(f"=== Attaching LoRA: {args.lora_path} ===")
        model = PeftModel.from_pretrained(model, args.lora_path)

    agent = AgentSimulator(
        model=model,
        processor=processor,
        system_prompt=system_prompt,
        sampling_config=sampling_config,
        device=args.device,
    )

    all_tasks = load_tasks()
    suite_ids = BENCHMARK_SUITES[args.suite]
    tasks = [t for t in all_tasks if t["instance_id"] in suite_ids]
    print(f"=== Loaded {len(tasks)} tasks for suite '{args.suite}' ===")

    results = []
    resolved_count = 0
    start_time = time.time()

    for idx, task in enumerate(tasks, 1):
        tid = task["instance_id"]
        repo = task.get("repo", "")
        print(f"\n--- [{idx}/{len(tasks)}] Task: {tid} ({repo}) ---")
        res = agent.solve_task(task, DATA_DIR, TEMP_BASE_DIR, timeout_sec=240, max_turns=20)
        status = res.get("status", "UNKNOWN")
        is_resolved = res.get("resolved", False)
        if is_resolved:
            resolved_count += 1
        dur = res.get("duration_sec", 0)
        turns = res.get("turns", 0)
        patch_sz = res.get("patch_size", 0)
        print(f"Result -> Status: {status} | Resolved: {is_resolved} | Duration: {dur}s | Turns: {turns} | Patch: {patch_sz}B")
        results.append(res)

    total_time = round(time.time() - start_time, 2)
    rate = (resolved_count / len(tasks)) * 100 if tasks else 0
    print("\n" + "="*50)
    print(f"Evaluation Complete: {resolved_count}/{len(tasks)} resolved ({rate:.1f}%) in {total_time}s")
    print("="*50)

if __name__ == "__main__":
    main()
