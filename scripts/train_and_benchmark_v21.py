#!/usr/bin/env python3
"""Automated pipeline to fine-tune, package, and benchmark v21 against original LoRA and Base."""
import os
import sys
import json
import time
import zipfile
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUTPUT_LORA_PATH = ROOT / "models" / "lora_v21_scottweeden_finetuned"
BUNDLE_V20 = ROOT / "state_of_the_art" / "bundle_v20"
OUT_ZIP = ROOT / "submissions_archive" / "submission_v21_lora.zip"

def run_step(cmd_list, desc):
    print(f"\n{'=' * 60}")
    print(f" STEP: {desc}")
    print(f" COMMAND: {' '.join(cmd_list)}")
    print(f"{'=' * 60}\n")
    start_t = time.time()
    ret = subprocess.run(cmd_list)
    dur = round(time.time() - start_t, 1)
    if ret.returncode != 0:
        print(f"ERROR: Step '{desc}' failed with exit code {ret.returncode}")
        sys.exit(ret.returncode)
    print(f"\n--> Step completed successfully in {dur}s.")

def package_submission():
    print(f"\n{'=' * 60}")
    print(f" STEP: Package v21 Submission Zip")
    print(f"{'=' * 60}\n")
    
    agent_yaml = """name: swe_agent
model: gemma-4-31b-it-qat-w4a16-ct
adapter: main_lora
instruction: !include prompts/system.md
tools:
  - run_command
  - read_file
  - edit_file
  - write_file
  - get_status
  - submit_patch
generate_content_config: !include configs/sampling.yaml
"""
    OUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in ["configs/sampling.yaml", "eval_config.yaml", "prompts/system.md"]:
            info = zipfile.ZipInfo(f, (1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            zf.writestr(info, (BUNDLE_V20 / f).read_bytes())
            
        info = zipfile.ZipInfo("agent.yaml", (1980, 1, 1, 0, 0, 0))
        info.external_attr = 0o100644 << 16
        zf.writestr(info, agent_yaml)
        
        for lora_file in ["adapter_config.json", "adapter_model.safetensors"]:
            info = zipfile.ZipInfo(f"adapters/main_lora/{lora_file}", (1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            zf.writestr(info, (OUTPUT_LORA_PATH / lora_file).read_bytes())
            
    print(f"Built {OUT_ZIP} (Size: {OUT_ZIP.stat().st_size / (1024*1024):.2f} MB)")
    
    # Run validator
    cmd = [
        "/home/lmassaron/code/venv_gemma4_12b/bin/python",
        str(ROOT / "scripts" / "build_v20_submission.py"),
        "--check",
        str(OUT_ZIP)
    ]
    subprocess.run(cmd, check=True)

def run_benchmark():
    eval_script = ROOT / "scripts" / "evaluate_v21_finetuned_lora.py"
    
    # Create evaluation script for the fine-tuned adapter
    eval_code = f"""#!/usr/bin/env python3
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
    lora_dir = "{str(OUTPUT_LORA_PATH)}"
    device = "cuda:0"
    
    print(f"Loading processor from {{base_model_path}}...")
    processor = AutoProcessor.from_pretrained(base_model_path)
    
    v20_dir = ROOT / "state_of_the_art" / "bundle_v20"
    prompt_v20 = (v20_dir / "prompts" / "system.md").read_text(encoding="utf-8")
    sampling_v20 = yaml.safe_load((v20_dir / "configs" / "sampling.yaml").read_text(encoding="utf-8"))
    
    print(f"Loading QAT model from {{base_model_path}} onto {{device}}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        device_map=device,
    )
    print(f"Attaching Fine-Tuned LoRA from {{lora_dir}}...")
    model = PeftModel.from_pretrained(model, lora_dir)
    print("Fine-tuned LoRA successfully attached to model!")
    
    agent = AgentSimulator(model, processor, prompt_v20, sampling_v20, device=device)
    
    print(f"\\n=======================================================")
    print(f" STARTING BENCHMARK: v21_finetuned_lora ({{len(tasks)}} tasks)")
    print(f"=======================================================")
    results = []
    resolved = 0
    start_t = time.time()
    
    for idx, t in enumerate(tasks, 1):
        tid = t["instance_id"]
        repo = t.get("repo", "")
        print(f"\\n[{{idx}}/{{len(tasks)}}] Task: {{tid}} ({{repo}})")
        res = agent.solve_task(t, DATA_DIR, TEMP_BASE_DIR, timeout_sec=240, max_turns=20)
        is_res = res.get("resolved", False)
        if is_res:
            resolved += 1
        dur = res.get("duration_sec", 0)
        turns = res.get("turns", 0)
        patch_sz = res.get("patch_size", 0)
        print(f"  -> Result: status={{res.get('status')}} | resolved={{is_res}} | dur={{dur}}s | turns={{turns}} | patch={{patch_sz}}B")
        results.append(res)
        
    tot_dur = round(time.time() - start_t, 2)
    rate = round((resolved / len(tasks)) * 100, 2) if tasks else 0
    print(f"\\n>>> v21_finetuned_lora Summary: {{resolved}}/{{len(tasks)}} resolved ({{rate}}%) in {{tot_dur}}s")
    
    out_file = ROOT / "state_of_the_art" / "benchmarks_data" / "benchmark_v21_finetuned_lora_core12_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({{
        "label": "v21_finetuned_lora",
        "resolved": resolved,
        "total": len(tasks),
        "pass_rate": rate,
        "total_duration": tot_dur,
        "tasks": results
    }}, indent=2), encoding="utf-8")
    print(f"\\nResults written to {{out_file}}")

if __name__ == "__main__":
    main()
"""
    eval_script.write_text(eval_code, encoding="utf-8")
    run_step(
        ["/home/lmassaron/code/venv_gemma4_12b/bin/python", str(eval_script)],
        "Benchmark Fine-Tuned v21 on core_12"
    )

def main():
    # 1. Run Fine-Tuning
    run_step(
        ["/home/lmassaron/code/venv_gemma4_12b/bin/python", str(ROOT / "scripts" / "finetune_scottweeden_lora.py")],
        "Fine-Tune Scott Weeden LoRA on Curated SWE Trajectories"
    )
    # 2. Package into submission_v21_lora.zip
    package_submission()
    # 3. Benchmark on core_12
    run_benchmark()
    print("\nALL PHASES (TRAIN -> PACKAGE -> BENCHMARK) COMPLETE!")

if __name__ == "__main__":
    main()
