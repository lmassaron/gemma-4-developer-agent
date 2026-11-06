#!/usr/bin/env python3
"""Patch and package a LoRA adapter submission for Kaggle.

Fixes the root causes of 0.00 LoRA scores on Kaggle:
1. Strips all multimodal (vision_tower / audio_tower) tensors that trigger
   `ValueError: Target module Gemma4ClippableLinear is not supported` in stock PEFT/vLLM.
2. Normalizes `adapter_config.json` to standard PEFT schema without bleeding-edge keys.
3. Sets `lora_alpha` to nominal scale (alpha / r <= 1.0) or custom scale.
4. Includes all 9 declared tools in `agent.yaml` and the proven system prompt.
5. Runs `check_submission.py` to assert zero rule failures.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parent.parent


def patch_adapter(src_adapter_dir: Path, out_adapter_dir: Path, target_scale: float = 1.0):
    out_adapter_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Clean safetensors
    sf_path = src_adapter_dir / "adapter_model.safetensors"
    assert sf_path.is_file(), f"Missing {sf_path}"
    
    with safe_open(str(sf_path), framework="pt", device="cpu") as sf:
        all_keys = list(sf.keys())
        cleaned_tensors = {}
        for k in all_keys:
            # Strictly language model layers only
            if "language_model.layers." in k and not any(bad in k.lower() for bad in ("vision", "audio", "embed")):
                cleaned_tensors[k] = sf.get_tensor(k).clone()
    
    print(f"[*] Filtered {len(all_keys)} adapter tensors down to {len(cleaned_tensors)} clean language tensors.")
    assert len(cleaned_tensors) in (820, 460, 410, 4), f"Unexpected tensor count: {len(cleaned_tensors)}"
    
    out_sf = out_adapter_dir / "adapter_model.safetensors"
    save_file(cleaned_tensors, str(out_sf))
    print(f"[OK] Saved cleaned safetensors: {out_sf} ({out_sf.stat().st_size / (1024*1024):.2f} MB)")
    
    # 2. Clean adapter_config.json
    cfg_path = src_adapter_dir / "adapter_config.json"
    assert cfg_path.is_file(), f"Missing {cfg_path}"
    src_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    
    r = int(src_cfg.get("r", 4))
    alpha = int(r * target_scale) if target_scale is not None else int(src_cfg.get("lora_alpha", r))
    
    target_modules = src_cfg.get("target_modules", [
        "down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"
    ])
    if len(cleaned_tensors) == 460:
        target_modules = ["k_proj", "o_proj", "q_proj", "v_proj"]
    
    clean_cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": src_cfg.get("base_model_name_or_path", "google/gemma-4-31b-it-qat-w4a16-ct"),
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": float(src_cfg.get("lora_dropout", 0.05)),
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "use_dora": False,
        "use_rslora": False,
        "modules_to_save": None,
        "target_modules": sorted(target_modules),
        "layers_to_transform": src_cfg.get("layers_to_transform", None),
        "layers_pattern": None,
        "rank_pattern": {},
        "alpha_pattern": {},
        "revision": None,
    }
    
    out_cfg = out_adapter_dir / "adapter_config.json"
    out_cfg.write_text(json.dumps(clean_cfg, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] Cleaned adapter_config.json: r={r}, lora_alpha={alpha} (scale={alpha/r:.2f})")


def build_submission(adapter_src: Path, out_zip: Path, target_scale: float = 1.0, max_time_minutes: int = 6, thinking_budget: int = 2048, fast_paced: bool = True):
    with tempfile.TemporaryDirectory() as td:
        staging_dir = Path(td) / "submission"
        staging_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Base configs & prompt from proven v26 / v23
        prompts_dir = staging_dir / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        configs_dir = staging_dir / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        
        # Read system.md from v26
        ref_v26_zip = REPO_ROOT / "submissions_archive" / "submission_v26_zero_shot_top_tier.zip"
        with zipfile.ZipFile(ref_v26_zip) as z:
            prompt_text = z.read("prompts/system.md").decode("utf-8")
        
        # Update minutes in prompt
        prompt_text = prompt_text.replace("8 minutes in total", f"{max_time_minutes} minutes in total")
        
        if fast_paced:
            old_plan = """# CALL PLAN & PACE
- Calls 1-8: locate and read.
- Calls 8-10: edit source code. You MUST have edited a source file by call 10 at the latest. If you are unsure, make your best-guess edit in the most likely place now. A wrong edit can still be corrected. Pure reading cannot score.
- Calls 10-14: verify with at most ONE focused command and submit immediately. Do NOT delay submission.
- Call 14 onward: no new exploration. Check `git status --short && git diff` and call `submit_patch` NOW.
- Always finish with `git status --short && git diff` followed by `submit_patch`."""
            new_plan = """# CALL PLAN & PACE
- Calls 1-5: locate and read.
- Call 6: edit source code. By call 6 at the latest, a source file MUST be edited. If unsure, make your best-guess edit in the most likely place now. A wrong edit can still be corrected. Pure reading cannot score.
- Calls 7-8: fast verify and immediate submit. Once a non-empty source edit is made, verify with at most ONE focused command and call submit_patch IMMEDIATELY. Do NOT delay submission. Never linger or explore after editing.
- Always finish with git status --short && git diff followed by submit_patch."""
            prompt_text = prompt_text.replace(old_plan, new_plan)
        
        (prompts_dir / "system.md").write_text(prompt_text, encoding="utf-8")
        
        sampling_yaml = f"""temperature: 0.2
top_p: 0.95
max_output_tokens: 8192
thinking_config:
  thinking_budget: {thinking_budget}
  include_thoughts: true
"""
        (configs_dir / "sampling.yaml").write_text(sampling_yaml, encoding="utf-8")
        
        eval_yaml = f"""evaluation:
  timeout_seconds: 240
  max_tool_calls: 28
  max_time_minutes: {max_time_minutes}
  max_turns: 80
"""
        (staging_dir / "eval_config.yaml").write_text(eval_yaml, encoding="utf-8")
        
        # 2. agent.yaml with all 9 tools and adapter declared
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
  - get_code_neighbors
  - search_similar_code
  - get_code_subgraph
generate_content_config: !include configs/sampling.yaml
"""
        (staging_dir / "agent.yaml").write_text(agent_yaml, encoding="utf-8")
        
        # 3. Patch and copy adapter
        patch_adapter(adapter_src, staging_dir / "adapters" / "main_lora", target_scale=target_scale)
        
        # 4. Create ZIP
        out_zip.parent.mkdir(parents=True, exist_ok=True)
        if out_zip.exists():
            out_zip.unlink()
        
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(staging_dir):
                for f in sorted(files):
                    fp = Path(root) / f
                    rel = fp.relative_to(staging_dir).as_posix()
                    zf.write(fp, rel)
        
        print(f"[OK] Built archive: {out_zip} ({out_zip.stat().st_size / (1024*1024):.2f} MB)")
        
        # 5. Run verification kit
        chk_script = Path("/tmp/dmitrii/check_submission.py")
        if chk_script.is_file():
            # Dmitrii checker strictly requires path to end in submission.zip
            chk_target = Path(td) / "submission.zip"
            shutil.copy2(out_zip, chk_target)
            proc = subprocess.run([sys.executable, str(chk_script), str(chk_target)], capture_output=True, text=True)
            print("\n[check_submission.py Output]:")
            print(proc.stdout.strip())
            assert proc.returncode == 0, f"check_submission failed: {proc.stdout}\n{proc.stderr}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-src", type=Path, required=True, help="Path to unpatched adapter dir")
    parser.add_argument("--out-zip", type=Path, required=True, help="Destination zip path")
    parser.add_argument("--scale", type=float, default=1.0, help="LoRA alpha/r scale (default: 1.0)")
    parser.add_argument("--max-time-minutes", type=int, default=6, help="Per-task time limit in minutes (default: 6)")
    parser.add_argument("--thinking-budget", type=int, default=2048, help="Thinking token budget (default: 2048)")
    parser.add_argument("--fast-paced", dest="fast_paced", action="store_true", default=True, help="Use fast-paced edit schedule (default: True)")
    parser.add_argument("--no-fast-paced", dest="fast_paced", action="store_false", help="Use classic 10-turn edit schedule")
    args = parser.parse_args()
    
    build_submission(args.adapter_src, args.out_zip, target_scale=args.scale, max_time_minutes=args.max_time_minutes, thinking_budget=args.thinking_budget, fast_paced=args.fast_paced)

