#!/usr/bin/env python3
"""
Package and validate the submission bundle for Kaggle Gemma 4 Developer Agent Competition.

Enforces competition rules:
1. Exactly one root config (agent.yaml or root_agent.yaml) at root.
2. Only allowed file extensions (.yaml, .yml, .md, .txt, .py, .json, .safetensors).
3. No symlinks.
4. Total uncompressed size < 3 GiB.
5. All '!include' references resolve properly.
6. Referenced adapters (if any) exist and have adapter_model.safetensors.
7. Valid allowed Gemma 4 model variant.
8. Deterministic ZIP packaging (sorted entries, fixed timestamps, file-only entries).
"""

import hashlib
import os
from pathlib import Path
import re
import shutil
import sys
import zipfile
import yaml

ROOT_CONFIGS = ["agent.yaml", "agent.yml", "root_agent.yaml", "root_agent.yml"]
ALLOWED_EXTS = {".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"}
ALLOWED_MODELS = {
    "gemma-4-31b-it-qat-w4a16-ct",
    "gemma-4-12b-it-qat-w4a16-ct",
    "gemma-4-e4b-it-qat-w4a16-ct",
    "gemma-4-e2b-it-qat-w4a16-ct",
}
MAX_UNCOMPRESSED_BYTES = 3 * 1024**3  # 3 GiB

class TolerantLoader(yaml.SafeLoader):
    pass

TolerantLoader.add_constructor("!include", lambda loader, node: loader.construct_scalar(node))

def validate_bundle(bundle_dir: Path) -> dict:
    bundle_dir = bundle_dir.resolve()
    assert bundle_dir.is_dir(), f"Bundle directory {bundle_dir} does not exist"

    files = [p for p in bundle_dir.rglob("*") if p.is_file()]
    assert files, "Bundle directory is empty"

    # 1. No symlinks
    symlinks = [p for p in bundle_dir.rglob("*") if p.is_symlink()]
    assert not symlinks, f"Symlinks are prohibited: {[str(p) for p in symlinks]}"

    # 2. Check root config
    roots = [name for name in ROOT_CONFIGS if (bundle_dir / name).is_file()]
    assert len(roots) == 1, f"Must have exactly one root config file, found {roots}"
    root_config = roots[0]

    # 3. Check allowed extensions
    invalid_exts = [p.name for p in files if p.suffix not in ALLOWED_EXTS]
    assert not invalid_exts, f"Files with invalid extensions found: {invalid_exts}"

    # 4. Check total uncompressed size
    total_size = sum(p.stat().st_size for p in files)
    assert total_size < MAX_UNCOMPRESSED_BYTES, f"Total size {total_size} exceeds 3 GiB"

    # 5. Check YAML syntax, !include targets, and model declarations
    models = set()
    for config_file in [p for p in files if p.suffix in (".yaml", ".yml")]:
        text = config_file.read_text(encoding="utf-8")
        data = yaml.load(text, Loader=TolerantLoader) or {}

        # Check !include references
        for include_target in re.findall(r"!include\s+(\S+)", text):
            target_path = config_file.parent / include_target
            assert target_path.is_file(), (
                f"{config_file.relative_to(bundle_dir)}: missing include {include_target}"
            )

        if "model" in data:
            models.add(data["model"])

        if "adapter" in data:
            adapter_name = data["adapter"]
            adapter_file = bundle_dir / "adapters" / adapter_name / "adapter_model.safetensors"
            assert adapter_file.is_file(), (
                f"{config_file.relative_to(bundle_dir)}: missing adapter {adapter_file}"
            )

    assert len(models) == 1, f"Must declare exactly one base model across submission, found {models}"
    model = models.pop()
    assert model in ALLOWED_MODELS, f"Model '{model}' is not in allowed Gemma 4 models: {ALLOWED_MODELS}"

    return {
        "root_config": root_config,
        "model": model,
        "file_count": len(files),
        "total_uncompressed_bytes": total_size,
    }

def create_deterministic_zip(bundle_dir: Path, output_zip: Path) -> Path:
    bundle_dir = bundle_dir.resolve()
    output_zip = output_zip.resolve()
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    output_zip.unlink(missing_ok=True)

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(p for p in bundle_dir.rglob("*") if p.is_file()):
            rel_name = p.relative_to(bundle_dir).as_posix()
            # File entries only, deterministic timestamp (1980-01-01)
            info = zipfile.ZipInfo(rel_name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, p.read_bytes())

    # Verify zip integrity
    with zipfile.ZipFile(output_zip) as z:
        assert z.testzip() is None, "ZIP file failed integrity test"
        names = z.namelist()
        assert "agent.yaml" in names or "root_agent.yaml" in names
        assert not any(n.endswith("/") for n in names), "ZIP contains directory entries!"

    return output_zip

def main():
    root = Path(__file__).resolve().parent.parent
    bundle_dir = root / "submission"
    output_zip = root / "submission.zip"

    print(f"Validating bundle at: {bundle_dir}")
    stats = validate_bundle(bundle_dir)
    print("Bundle validation passed:")
    for k, v in stats.items():
        print(f"  - {k}: {v}")

    print(f"\nBuilding deterministic ZIP at: {output_zip}")
    zip_path = create_deterministic_zip(bundle_dir, output_zip)
    sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    print(f"Package created: {zip_path.name}")
    print(f"  Size: {zip_path.stat().st_size:,} bytes")
    print(f"  SHA-256: {sha256}")

    # Test roundtrip
    test_unpack = root / ".test_unpack"
    shutil.rmtree(test_unpack, ignore_errors=True)
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(test_unpack)
        validate_bundle(test_unpack)
        print("Roundtrip extract & validation: OK")
    finally:
        shutil.rmtree(test_unpack, ignore_errors=True)

    print("\nSUCCESS: submission.zip is 100% compliant and ready for Kaggle!")

if __name__ == "__main__":
    main()
