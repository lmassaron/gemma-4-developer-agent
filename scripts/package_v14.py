#!/usr/bin/env python3
"""
Build and validate submission_v14_heredoc_triage_synthesis.zip.
"""
import sys
from pathlib import Path
import hashlib
import zipfile
import shutil
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from scripts.package_and_validate import validate_bundle, create_deterministic_zip
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from package_and_validate import validate_bundle, create_deterministic_zip


def main():
    root = Path(__file__).resolve().parent.parent
    bundle_dir = root / "state_of_the_art" / "bundle_v14"
    output_zip = root / "submissions_archive" / "submission_v14_heredoc_triage_synthesis.zip"

    print(f"1. Validating bundle at: {bundle_dir}")
    stats = validate_bundle(bundle_dir)
    for k, v in stats.items():
        print(f"   - {k}: {v}")

    print(f"\n2. Creating deterministic ZIP at: {output_zip}")
    zip_path = create_deterministic_zip(bundle_dir, output_zip)
    sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    size_bytes = zip_path.stat().st_size
    print(f"   - Filename: {zip_path.name}")
    print(f"   - Size: {size_bytes:,} bytes")
    print(f"   - SHA-256: {sha256}")

    print("\n3. Testing ZIP roundtrip & extraction:")
    test_unpack = root / ".test_unpack_v14"
    shutil.rmtree(test_unpack, ignore_errors=True)
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            print(f"   - Archived files ({len(names)}): {names}")
            assert "agent.yaml" in names, "Missing agent.yaml in root"
            assert "eval_config.yaml" in names, "Missing eval_config.yaml in root"
            assert "configs/sampling.yaml" in names, "Missing configs/sampling.yaml"
            assert "prompts/system.md" in names, "Missing prompts/system.md"
            z.extractall(test_unpack)
        validate_bundle(test_unpack)
        print("   - Integrity test: PASSED (100% compliant with Kaggle specifications)")
    finally:
        shutil.rmtree(test_unpack, ignore_errors=True)

    print("\nSUCCESS: submission_v14_heredoc_triage_synthesis.zip is ready!")

if __name__ == "__main__":
    main()
