#!/usr/bin/env python3
"""
CPU-Only Background Filter for External HuggingFace SWE-bench Datasets (Python 3.12 Compatible).

Scans HuggingFace datasets (`princeton-nlp/SWE-bench_Verified`, `princeton-nlp/SWE-bench_Lite`,
and `SWE-bench/SWE-smith`) across diverse Python repositories (`pytest-dev/pytest`, `pallets/flask`,
`psf/requests`, `pylint-dev/pylint`, `sympy/sympy`, `sphinx-doc/sphinx`, `pydata/xarray`),
tests each task in our local Python 3.12 environment (`TaskWorkspace`), and saves every task where:
  1. `before_gold == False` (the unpatched base_commit fails the hidden test_patch)
  2. `after_gold == True`   (applying the official gold `patch` passes the hidden test_patch in Python 3.12)
into `data/external_hf_benchmark_py312.jsonl`.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
scripts_dir = str(Path(__file__).resolve().parent)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

# Force CPU-only so we never touch the GPU while hill_climb_v18.py is running!
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from datasets import load_dataset
from run_local_benchmark_leaderboard import TaskWorkspace


TARGET_REPOS = (
    "django/django",
    "scikit-learn/scikit-learn",
    "pydata/xarray",
    "astropy/astropy",
    "matplotlib/matplotlib",
    "sympy/sympy",
    "sphinx-doc/sphinx",
    "pytest-dev/pytest",
    "pallets/flask",
    "psf/requests",
    "pylint-dev/pylint",
)


def verify_task_py312(task: dict, data_dir: Path, work_root: Path) -> bool:
    ws = TaskWorkspace(task, data_dir, work_root)
    try:
        ws.setup()
        # 1. Verify test fails BEFORE gold patch
        passed_before, _ = ws.run_verification(timeout_sec=25)
        if passed_before:
            return False

        # 2. Reset working tree so hidden test_patch can be cleanly re-applied after gold patch!
        subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=ws.work_dir, capture_output=True)
        subprocess.run(["git", "clean", "-fd"], cwd=ws.work_dir, capture_output=True)

        # 3. Apply official gold patch
        gold_patch = task.get("patch", "")
        if not gold_patch.strip():
            return False
        pf = ws.work_dir / ".gold.patch"
        clean_gold = gold_patch if gold_patch.endswith("\n") else (gold_patch + "\n")
        pf.write_text(clean_gold, encoding="utf-8")
        res = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", str(pf)],
            cwd=ws.work_dir,
            capture_output=True,
            text=True,
        )
        pf.unlink(missing_ok=True)
        if res.returncode != 0:
            return False

        # 4. Verify test passes AFTER gold patch in Python 3.12
        passed_after, _ = ws.run_verification(timeout_sec=25)
        return passed_after
    except Exception:
        return False
    finally:
        ws.cleanup()


def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    out_file = data_dir / "external_hf_benchmark_py312.jsonl"
    work_root = Path("/tmp/hf_external_filter_workspaces")
    work_root.mkdir(parents=True, exist_ok=True)

    existing_ids = set()
    if out_file.is_file():
        for line in out_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing_ids.add(json.loads(line)["instance_id"])

    print(f"[HF External Filter] Loading princeton-nlp/SWE-bench_Verified (CPU-only)...", flush=True)
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    # Sort by instance_id descending within each repo so we test newer commits (more likely Python 3.12 compatible) first!
    candidates = [dict(row) for row in ds if row["repo"] in TARGET_REPOS]
    candidates.sort(key=lambda x: x["instance_id"], reverse=True)

    print(f"[HF External Filter] Found {len(candidates)} candidate tasks across {TARGET_REPOS}. Testing gold patches...", flush=True)
    verified_count = len(existing_ids)
    per_repo_counts = {}
    per_repo_tested = {}
    if out_file.is_file():
        for line in out_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)["repo"]
                per_repo_counts[r] = per_repo_counts.get(r, 0) + 1

    for idx, t in enumerate(candidates, 1):
        tid = t["instance_id"]
        repo = t["repo"]
        if tid in existing_ids:
            continue
        if per_repo_counts.get(repo, 0) >= 12 or per_repo_tested.get(repo, 0) >= 60:
            continue
        per_repo_tested[repo] = per_repo_tested.get(repo, 0) + 1

        ok = verify_task_py312(t, data_dir, work_root)
        status_str = "PASS (Added!)" if ok else "skip (legacy py env)"
        print(f"  [{idx}/{len(candidates)}] {tid:<28} ({repo}) -> {status_str}", flush=True)
        if ok:
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(t) + "\n")
            existing_ids.add(tid)
            per_repo_counts[repo] = per_repo_counts.get(repo, 0) + 1
            verified_count += 1
            if verified_count >= 20:
                break

    print(f"\n[HF External Filter] Done! Saved {verified_count} verified Python 3.12 external tasks to {out_file}", flush=True)
    print(f"  Per-repo breakdown: {per_repo_counts}", flush=True)


if __name__ == "__main__":
    main()

