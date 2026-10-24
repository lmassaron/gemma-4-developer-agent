#!/usr/bin/env python3
"""
Package and validate the Gemma 4 12B Prompt-Only submission bundle for Kaggle.

Produces: submission_gemma4_12b_prompt.zip & submissions_archive/submission_v5_gemma4_12b_prompt.zip
Includes:
- Root agent.yaml declaring:
    model: gemma-4-12b-it-qat-w4a16-ct
- prompts/system.md
- configs/sampling.yaml
- eval_config.yaml
"""

import hashlib
import os
from pathlib import Path
import shutil
import sys
import zipfile
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
STAGING_DIR = ROOT_DIR / ".staging_12b_prompt_submission"
OUTPUT_ZIP = ROOT_DIR / "submission_gemma4_12b_prompt.zip"
ARCHIVE_ZIP = ROOT_DIR / "submissions_archive" / "submission_v5_gemma4_12b_prompt.zip"

ALLOWED_EXTS = {".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"}
MAX_UNCOMPRESSED_BYTES = 3 * 1024**3  # 3 GiB

def assemble_staging_bundle():
    print("[1/4] Assembling 12B prompt staging bundle...")
    shutil.rmtree(STAGING_DIR, ignore_errors=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Base files from submission/
    shutil.copytree(ROOT_DIR / "submission" / "configs", STAGING_DIR / "configs")
    shutil.copytree(ROOT_DIR / "submission" / "prompts", STAGING_DIR / "prompts")
    shutil.copy(ROOT_DIR / "submission" / "eval_config.yaml", STAGING_DIR / "eval_config.yaml")

    # 2. Agent.yaml with 12B model reference
    agent_yaml_content = """name: swe_agent
model: gemma-4-12b-it-qat-w4a16-ct
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

    root_config = bundle_dir / "agent.yaml"
    assert root_config.is_file(), "Missing agent.yaml at bundle root"

    return {
        "file_count": len(files),
        "total_uncompressed_bytes": total_size,
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
    assemble_staging_bundle()
    stats = validate_bundle(STAGING_DIR)
    zip_path = create_deterministic_zip(STAGING_DIR, OUTPUT_ZIP)

    # Also copy to submissions_archive
    ARCHIVE_ZIP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(zip_path, ARCHIVE_ZIP)

    sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    print(f"\n[4/4] SUCCESS: Package created: {zip_path.name}")
    print(f"  - Location: {zip_path}")
    print(f"  - Copied to archive: {ARCHIVE_ZIP}")
    print(f"  - Compressed Size: {zip_path.stat().st_size:,} bytes")
    print(f"  - Uncompressed Size: {stats['total_uncompressed_bytes']:,} bytes")
    print(f"  - SHA-256: {sha256}")

    # Clean up staging dir
    shutil.rmtree(STAGING_DIR, ignore_errors=True)

if __name__ == "__main__":
    main()
