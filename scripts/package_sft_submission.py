#!/usr/bin/env python3
"""
Package and validate the Trajectory-SFT LoRA submission bundle for Kaggle.

Produces: submission_gemma4_31b_sft.zip
Includes:
- Root agent.yaml declaring:
    model: gemma-4-31b-it-qat-w4a16-ct
    adapter: gemma4_31b_swe_lora
- adapters/gemma4_31b_swe_lora/
    - adapter_config.json
    - adapter_model.safetensors (232 MB trained LoRA weights for 60 layers)
- prompts/system.md
- configs/sampling.yaml
- eval_config.yaml
"""

import os
import sys
import shutil
import hashlib
import zipfile
import re
from pathlib import Path
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
STAGING_DIR = ROOT_DIR / ".staging_sft_submission"
OUTPUT_ZIP = ROOT_DIR / "submission_gemma4_31b_sft.zip"
ADAPTER_SRC = ROOT_DIR / "adapters" / "gemma4_31b_swe_lora"

ALLOWED_EXTS = {".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"}
MAX_UNCOMPRESSED_BYTES = 3 * 1024**3  # 3 GiB

class TolerantLoader(yaml.SafeLoader):
    pass
TolerantLoader.add_constructor("!include", lambda loader, node: loader.construct_scalar(node))

def assemble_staging_bundle(adapter_dir: Path):
    print(f"[1/4] Assembling staging bundle from {adapter_dir}...")
    shutil.rmtree(STAGING_DIR, ignore_errors=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Base files from submission/
    shutil.copytree(ROOT_DIR / "submission" / "configs", STAGING_DIR / "configs")
    shutil.copytree(ROOT_DIR / "submission" / "prompts", STAGING_DIR / "prompts")
    shutil.copy(ROOT_DIR / "submission" / "eval_config.yaml", STAGING_DIR / "eval_config.yaml")

    # 2. Agent.yaml with adapter reference
    agent_yaml_content = """name: swe_agent
model: gemma-4-31b-it-qat-w4a16-ct
adapter: gemma4_31b_swe_lora
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
    (STAGING_DIR / "agent.yaml").write_text(agent_yaml_content, encoding="utf-8")

    # 3. Copy adapter files
    target_adapter_dir = STAGING_DIR / "adapters" / "gemma4_31b_swe_lora"
    target_adapter_dir.mkdir(parents=True, exist_ok=True)

    cfg_file = adapter_dir / "adapter_config.json"
    weights_file = adapter_dir / "adapter_model.safetensors"

    assert cfg_file.is_file(), f"Missing {cfg_file}"
    assert weights_file.is_file(), f"Missing {weights_file}"

    shutil.copy(cfg_file, target_adapter_dir / "adapter_config.json")
    shutil.copy(weights_file, target_adapter_dir / "adapter_model.safetensors")
    print(f"  Copied adapter files ({weights_file.stat().st_size / (1024*1024):.2f} MB)")

def validate_bundle(bundle_dir: Path) -> dict:
    print("[2/4] Validating bundle compliance with Kaggle competition rules...")
    files = [p for p in bundle_dir.rglob("*") if p.is_file()]
    assert files, "Bundle is empty"

    # Symlinks check
    symlinks = [p for p in bundle_dir.rglob("*") if p.is_symlink()]
    assert not symlinks, f"Symlinks prohibited: {symlinks}"

    # Extension check
    invalid_exts = [p.name for p in files if p.suffix not in ALLOWED_EXTS]
    assert not invalid_exts, f"Invalid file extensions: {invalid_exts}"

    # Size check
    total_size = sum(p.stat().st_size for p in files)
    assert total_size < MAX_UNCOMPRESSED_BYTES, f"Total size {total_size} exceeds 3 GiB"

    # Adapter verification
    adapter_weights = bundle_dir / "adapters" / "gemma4_31b_swe_lora" / "adapter_model.safetensors"
    assert adapter_weights.is_file(), "Missing adapter_model.safetensors in bundle"

    root_config = bundle_dir / "agent.yaml"
    assert root_config.is_file(), "Missing agent.yaml at bundle root"

    return {
        "file_count": len(files),
        "total_uncompressed_bytes": total_size,
        "adapter_size_mb": adapter_weights.stat().st_size / (1024*1024),
    }

def create_deterministic_zip(bundle_dir: Path, output_zip: Path) -> Path:
    print(f"[3/4] Building deterministic zip at {output_zip}...")
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    output_zip.unlink(missing_ok=True)

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(p for p in bundle_dir.rglob("*") if p.is_file()):
            rel_name = p.relative_to(bundle_dir).as_posix()
            info = zipfile.ZipInfo(rel_name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, p.read_bytes())

    with zipfile.ZipFile(output_zip) as z:
        assert z.testzip() is None, "Corrupted zip file"
        names = z.namelist()
        assert "agent.yaml" in names, "agent.yaml missing from zip root"
        assert not any(n.endswith("/") for n in names), "Directory entries found in zip"

    return output_zip

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", type=str, default=str(ADAPTER_SRC), help="Path to trained 31B LoRA adapter")
    args = parser.parse_args()

    adapter_path = Path(args.adapter_dir).resolve()
    if not adapter_path.is_dir():
        print(f"Error: Adapter directory {adapter_path} does not exist yet.")
        sys.exit(1)

    assemble_staging_bundle(adapter_path)
    stats = validate_bundle(STAGING_DIR)
    zip_path = create_deterministic_zip(STAGING_DIR, OUTPUT_ZIP)

    sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    print(f"\n[4/4] SUCCESS: Package created: {zip_path.name}")
    print(f"  - Location: {zip_path}")
    print(f"  - Compressed Size: {zip_path.stat().st_size / (1024*1024):.2f} MB ({zip_path.stat().st_size:,} bytes)")
    print(f"  - Uncompressed Size: {stats['total_uncompressed_bytes'] / (1024*1024):.2f} MB")
    print(f"  - Adapter Size: {stats['adapter_size_mb']:.2f} MB")
    print(f"  - SHA-256: {sha256}")

    # Clean up staging dir
    shutil.rmtree(STAGING_DIR, ignore_errors=True)

if __name__ == "__main__":
    main()
