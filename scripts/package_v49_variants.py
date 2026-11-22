#!/usr/bin/env python3
"""Package zero-retrain conditional variants of v47 and v49 for fast Kaggle iteration.

Produces (and validates via data/dmitrii_kit/check_submission.py):
  1. submission_v49_scale075.zip  — v49 Edit-Span DPO adapter rescaled to 0.75x (effective scale 0.75)
  2. submission_v49_tb3072.zip    — v49 Edit-Span DPO with thinking_budget: 3072 (12h runtime cushion)
  3. submission_v47_tb3072.zip    — v47 zero-shot with thinking_budget: 3072 (12h runtime cushion)
"""
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from package_v47_candidate import V47_AGENT_YAML, V47_EVAL_YAML, V47_SYSTEM_MD
from train_v48_edit_only_lora import V48_AGENT_YAML, V48_SAMPLING_YAML

V49_ADAPTER_DIR = ROOT / "models" / "lora_v49_edit_dpo"
V49_SCALE075_DIR = ROOT / "models" / "lora_v49_edit_dpo_scale075"

SAMPLING_TB3072_YAML = """temperature: 0.2
top_p: 0.95
max_output_tokens: 6144
thinking_config:
  thinking_budget: 3072
  include_thoughts: true
"""


def write_zip(out_zip: Path, text_files: list[tuple[str, str]], adapter_dir: Path | None = None) -> str:
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel, content in sorted(text_files):
            info = zipfile.ZipInfo(rel, (1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            zf.writestr(info, content.encode("utf-8"))
        if adapter_dir is not None:
            zf.write(adapter_dir / "adapter_config.json", arcname="adapters/main_lora/adapter_config.json")
            zf.write(adapter_dir / "adapter_model.safetensors", arcname="adapters/main_lora/adapter_model.safetensors")
    return hashlib.sha256(out_zip.read_bytes()).hexdigest()


def build_scaled_adapter(src_dir: Path, dst_dir: Path, scale_mult: float) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    state = load_safetensors(str(src_dir / "adapter_model.safetensors"))
    scaled = {}
    n_b = 0
    for k, v in state.items():
        if k.endswith(".lora_B.weight"):
            scaled[k] = (v.float() * scale_mult).to(torch.bfloat16).contiguous()
            n_b += 1
        else:
            scaled[k] = v.contiguous()
    assert n_b == 90, f"Expected 90 lora_B tensors, got {n_b}"
    save_safetensors(scaled, str(dst_dir / "adapter_model.safetensors"))
    shutil.copy2(src_dir / "adapter_config.json", dst_dir / "adapter_config.json")


def validate_zip(zip_path: Path) -> None:
    tmp_sub = Path("/tmp/submission.zip")
    shutil.copy2(zip_path, tmp_sub)
    res = subprocess.run(
        [sys.executable, str(ROOT / "data" / "dmitrii_kit" / "check_submission.py"), str(tmp_sub)],
        capture_output=True,
        text=True,
        check=False,
    )
    print(f"--- check_submission: {zip_path.name} ---")
    print(res.stdout.strip())
    if res.returncode != 0:
        print(res.stderr)
        sys.exit(res.returncode)


def main():
    assert V49_ADAPTER_DIR.is_dir(), f"Missing {V49_ADAPTER_DIR}"

    # 1. v49 scaled to 0.75x (effective scale 0.75)
    build_scaled_adapter(V49_ADAPTER_DIR, V49_SCALE075_DIR, scale_mult=0.75)
    z1 = ROOT / "submission_v49_scale075.zip"
    sha1 = write_zip(
        z1,
        [
            ("agent.yaml", V48_AGENT_YAML),
            ("configs/sampling.yaml", V48_SAMPLING_YAML),
            ("eval_config.yaml", V47_EVAL_YAML),
            ("prompts/system.md", V47_SYSTEM_MD),
        ],
        adapter_dir=V49_SCALE075_DIR,
    )
    print(f"[OK] {z1.name} ({z1.stat().st_size / (1024*1024):.2f} MB) | SHA256={sha1}")
    validate_zip(z1)

    # 2. v49 with thinking_budget=3072
    z2 = ROOT / "submission_v49_tb3072.zip"
    sha2 = write_zip(
        z2,
        [
            ("agent.yaml", V48_AGENT_YAML),
            ("configs/sampling.yaml", SAMPLING_TB3072_YAML),
            ("eval_config.yaml", V47_EVAL_YAML),
            ("prompts/system.md", V47_SYSTEM_MD),
        ],
        adapter_dir=V49_ADAPTER_DIR,
    )
    print(f"[OK] {z2.name} ({z2.stat().st_size / (1024*1024):.2f} MB) | SHA256={sha2}")
    validate_zip(z2)

    # 3. v47 zero-shot with thinking_budget=3072
    z3 = ROOT / "submission_v47_tb3072.zip"
    sha3 = write_zip(
        z3,
        [
            ("agent.yaml", V47_AGENT_YAML),
            ("configs/sampling.yaml", SAMPLING_TB3072_YAML),
            ("eval_config.yaml", V47_EVAL_YAML),
            ("prompts/system.md", V47_SYSTEM_MD),
        ],
        adapter_dir=None,
    )
    print(f"[OK] {z3.name} ({z3.stat().st_size} B) | SHA256={sha3}")
    validate_zip(z3)


if __name__ == "__main__":
    main()
