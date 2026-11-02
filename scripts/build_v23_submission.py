#!/usr/bin/env python3
"""
Build and validate the v23 Zero-Shot SOTA submission bundle for Kaggle.
Pure prompt engineering on gemma-4-31b-it-qat-w4a16-ct without LoRA adapters.
100% compliant with Kaggle ADK, zero KV-cache overhead, size ~6.5 KB.
"""

import hashlib
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "state_of_the_art" / "bundle_v23_zero_shot"
OUT_ZIP = ROOT / "submissions_archive" / "submission_v23_zero_shot_sota.zip"
ROOT_ZIP = ROOT / "submission.zip"

EXPECTED_FILES = [
    "agent.yaml",
    "eval_config.yaml",
    "configs/sampling.yaml",
    "prompts/system.md",
]


def validate_source():
    assert BUNDLE_DIR.is_dir(), f"Missing bundle directory {BUNDLE_DIR}"
    for rel in EXPECTED_FILES:
        p = BUNDLE_DIR / rel
        assert p.is_file(), f"Missing expected file: {p}"
    
    # Check braces in system prompt
    prompt_text = (BUNDLE_DIR / "prompts" / "system.md").read_text(encoding="utf-8")
    assert "{" not in prompt_text and "}" not in prompt_text, (
        "Unescaped braces detected in prompts/system.md!"
    )
    print("Bundle source pre-checks passed successfully.")


def build():
    validate_source()
    OUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()

    print(f"Packaging {OUT_ZIP}...")
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in sorted(EXPECTED_FILES):
            p = BUNDLE_DIR / rel
            info = zipfile.ZipInfo(rel, (1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            zf.writestr(info, p.read_bytes())
            print(f"  + Added {rel} ({p.stat().st_size} bytes)")

    size_bytes = OUT_ZIP.stat().st_size
    sha256 = hashlib.sha256(OUT_ZIP.read_bytes()).hexdigest()
    print(f"Successfully packaged {OUT_ZIP.name}")
    print(f"  Size: {size_bytes:,} bytes ({size_bytes / 1024:.2f} KB)")
    print(f"  SHA-256: {sha256}")

    # Copy to root submission.zip
    shutil.copy2(OUT_ZIP, ROOT_ZIP)
    print(f"Updated root {ROOT_ZIP.name}")

    # Run strict packaging validator
    val_script = ROOT / "scripts" / "package_and_validate.py"
    if val_script.is_file():
        import subprocess
        res = subprocess.run([sys.executable, str(val_script), "--verify-only", str(OUT_ZIP)], capture_output=True, text=True)
        print(res.stdout)
        if res.returncode != 0:
            print("STDERR:", res.stderr)
            raise RuntimeError(f"Validation failed with code {res.returncode}")


if __name__ == "__main__":
    build()
