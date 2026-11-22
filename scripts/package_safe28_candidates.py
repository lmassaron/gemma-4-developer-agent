#!/usr/bin/env python3
"""Package 12h-safe (28-call / 8m / max_output_tokens=6144 / tb=4096) zips for v47, v48, and v49.

Why:
  - v47 (40 calls / 8m / tb=4096 / max_out=8192) hit Kaggle's 12h Notebook Timeout on ref 58637088.
  - All 7 prior 28-call/8m/tb=4096 submissions (v23, v37, v40, v42, v45, v46 183MB LoRA, exp021)
    AND ayush04barai's 0.22 90-module LoRA (28 calls / 8m / max_out=6144 / tb=4096) completed cleanly.
  - Preserves v47's compaction-pinned {problem_description}/{hints?} + fast exit + v48/v49 LoRAs.
"""
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from package_v47_candidate import V47_AGENT_YAML, V47_SYSTEM_MD
from train_v48_edit_only_lora import V48_AGENT_YAML, V48_SAMPLING_YAML

SAFE28_EVAL_YAML = """evaluation:
  timeout_seconds: 60
  max_tool_calls: 28
  max_time_minutes: 8
  max_turns: 80
"""

MID34_EVAL_YAML = """evaluation:
  timeout_seconds: 60
  max_tool_calls: 34
  max_time_minutes: 7.5
  max_turns: 80
"""

TB3584_SAMPLING_YAML = """temperature: 0.2
top_p: 0.95
max_output_tokens: 6144
thinking_config:
  thinking_budget: 3584
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
    targets = [
        ("submission_v49_safe28.zip", V48_AGENT_YAML, V48_SAMPLING_YAML, SAFE28_EVAL_YAML, ROOT / "models" / "lora_v49_edit_dpo"),
        ("submission_v49_mid34.zip", V48_AGENT_YAML, TB3584_SAMPLING_YAML, MID34_EVAL_YAML, ROOT / "models" / "lora_v49_edit_dpo"),
        ("submission_v48_safe28.zip", V48_AGENT_YAML, V48_SAMPLING_YAML, SAFE28_EVAL_YAML, ROOT / "models" / "lora_v48_edit_only"),
        ("submission_v47_safe28.zip", V47_AGENT_YAML, V48_SAMPLING_YAML, SAFE28_EVAL_YAML, None),
    ]
    for zname, agent_yaml, sampling_yaml, eval_yaml, adir in targets:
        if adir is not None:
            assert (adir / "adapter_model.safetensors").is_file(), f"Missing adapter in {adir}"
        zpath = ROOT / zname
        sha = write_zip(
            zpath,
            [
                ("agent.yaml", agent_yaml),
                ("configs/sampling.yaml", sampling_yaml),
                ("eval_config.yaml", eval_yaml),
                ("prompts/system.md", V47_SYSTEM_MD),
            ],
            adapter_dir=adir,
        )
        size_str = f"{zpath.stat().st_size / (1024*1024):.2f} MB" if adir else f"{zpath.stat().st_size} B"
        print(f"[OK] {zname} ({size_str}) | SHA256={sha}")
        validate_zip(zpath)


if __name__ == "__main__":
    main()
