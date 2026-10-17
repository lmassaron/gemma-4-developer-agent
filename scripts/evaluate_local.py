#!/usr/bin/env python3
"""
Local SWE-bench Benchmark & Evaluation Harness for Gemma 4 Developer Agent Competition.

Features:
- Extracts snapshot from data/snapshots/<instance_id>.tgz into isolated test workspace
- Automatically extracts target test files and specific test functions added by test_patch
- Accurately runs targeted pytest with negative/positive control testing
- Supports:
  1. 'audit': Verifies task soundness (empty patch fails, reference patch passes)
  2. 'gold-eval': Evaluates reference solutions to baseline the benchmark
  3. 'patch-eval': Evaluates a user-provided patch or agent patch
- Computes PASS / FAIL metrics and Resolution Rate
- Exports results to evaluation_results.json
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

def extract_target_test_info(test_patch: str) -> tuple[list[str], list[str]]:
    """Extract paths to test files and test function names added in test_patch."""
    files = []
    functions = []
    for line in test_patch.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:].strip()
            if path != "/dev/null":
                files.append(path)
        elif line.startswith("+") and not line.startswith("+++"):
            m = re.search(r"def\s+(test_[a-zA-Z0-9_]+)", line)
            if m:
                functions.append(m.group(1))
    return list(dict.fromkeys(files)), list(dict.fromkeys(functions))

class TaskWorkspace:
    def __init__(self, task: dict, base_dir: Path, data_dir: Path):
        self.task = task
        self.instance_id = task["instance_id"]
        self.data_dir = data_dir
        self.work_dir = base_dir / self.instance_id
        self.snapshot_tar = data_dir / "snapshots" / f"{self.instance_id}.tgz"

    def setup(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        if not self.snapshot_tar.is_file():
            raise FileNotFoundError(f"Snapshot not found: {self.snapshot_tar}")

        with tarfile.open(self.snapshot_tar, "r:gz") as tar:
            tar.extractall(self.work_dir)

        # Initialize git repo
        if not (self.work_dir / ".git").is_dir():
            subprocess.run(["git", "init", "-q"], cwd=self.work_dir, check=True)
            subprocess.run(["git", "config", "user.name", "evaluator"], cwd=self.work_dir, check=True)
            subprocess.run(["git", "config", "user.email", "eval@kaggle.local"], cwd=self.work_dir, check=True)
            subprocess.run(["git", "add", "-A"], cwd=self.work_dir, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "base snapshot", "--allow-empty"], cwd=self.work_dir, check=True)

    def apply_patch(self, patch_text: str) -> tuple[bool, str]:
        if not patch_text or not patch_text.strip():
            return True, ""
        if not patch_text.endswith("\n"):
            patch_text += "\n"

        patch_file = self.work_dir / ".tmp_eval.patch"
        patch_file.write_text(patch_text, encoding="utf-8")
        try:
            res = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", str(patch_file)],
                cwd=self.work_dir,
                capture_output=True,
                text=True
            )
            return (res.returncode == 0, res.stderr)
        finally:
            patch_file.unlink(missing_ok=True)

    def run_tests(self, timeout_sec: int = 60) -> tuple[bool, str]:
        test_files, test_functions = extract_target_test_info(self.task.get("test_patch", ""))
        pytest_cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short"]
        if test_files:
            pytest_cmd.extend(test_files)
        if test_functions:
            expr = " or ".join(test_functions)
            pytest_cmd.extend(["-k", expr])

        env = os.environ.copy()
        python_paths = [str(self.work_dir), str(self.work_dir / "src")]
        if env.get("PYTHONPATH"):
            python_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = ":".join(python_paths)

        try:
            proc = subprocess.run(
                pytest_cmd,
                cwd=self.work_dir,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                env=env
            )
            output = proc.stdout + "\n" + proc.stderr
            return (proc.returncode == 0, output)
        except subprocess.TimeoutExpired:
            return (False, f"Test timed out after {timeout_sec}s")
        except Exception as e:
            return (False, f"Error running tests: {e}")

    def cleanup(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

def evaluate_task_soundness(task: dict, data_dir: Path, temp_dir: Path) -> dict:
    """Verify task soundness: empty patch fails tests, reference patch passes tests."""
    ws = TaskWorkspace(task, temp_dir, data_dir)
    ws.setup()
    try:
        # Negative control
        ok_test, err_test = ws.apply_patch(task["test_patch"])
        if not ok_test:
            return {"instance_id": task["instance_id"], "repo": task["repo"], "status": "TEST_PATCH_FAILED_TO_APPLY", "error": err_test}

        passed_empty, out_empty = ws.run_tests(timeout_sec=30)

        # Reset workspace for positive control
        ws.setup()
        ok_gold, err_gold = ws.apply_patch(task["patch"])
        if not ok_gold:
            return {"instance_id": task["instance_id"], "repo": task["repo"], "status": "GOLD_PATCH_FAILED_TO_APPLY", "error": err_gold}

        ok_test, err_test = ws.apply_patch(task["test_patch"])
        if not ok_test:
            return {"instance_id": task["instance_id"], "repo": task["repo"], "status": "TEST_PATCH_CONFLICTS_WITH_GOLD", "error": err_test}

        passed_gold, out_gold = ws.run_tests(timeout_sec=30)

        if not passed_empty and passed_gold:
            status = "SOUND"
        elif passed_empty:
            status = "PASSES_WITHOUT_FIX"
        else:
            status = "GOLD_FAILS_TESTS"

        return {
            "instance_id": task["instance_id"],
            "repo": task["repo"],
            "status": status,
            "sound": status == "SOUND",
            "passed_empty": passed_empty,
            "passed_gold": passed_gold,
        }
    finally:
        ws.cleanup()

def evaluate_patch(task: dict, patch: str, data_dir: Path, temp_dir: Path) -> dict:
    """Evaluate a patch against the verification test suite."""
    ws = TaskWorkspace(task, temp_dir, data_dir)
    ws.setup()
    t0 = time.time()
    try:
        ok_patch, err_patch = ws.apply_patch(patch)
        if not ok_patch:
            return {
                "instance_id": task["instance_id"],
                "repo": task["repo"],
                "status": "PATCH_APPLY_ERROR",
                "resolved": False,
                "duration_sec": round(time.time() - t0, 2),
                "error": err_patch,
            }

        ok_test, err_test = ws.apply_patch(task["test_patch"])
        if not ok_test:
            return {
                "instance_id": task["instance_id"],
                "repo": task["repo"],
                "status": "TEST_PATCH_CONFLICT",
                "resolved": False,
                "duration_sec": round(time.time() - t0, 2),
                "error": err_test,
            }

        passed, out = ws.run_tests(timeout_sec=45)
        return {
            "instance_id": task["instance_id"],
            "repo": task["repo"],
            "status": "RESOLVED" if passed else "FAILED",
            "resolved": passed,
            "duration_sec": round(time.time() - t0, 2),
            "output_snippet": out[-500:] if out else "",
        }
    finally:
        ws.cleanup()

def main():
    parser = argparse.ArgumentParser(description="Local SWE Benchmark Evaluator")
    parser.add_argument("--tasks", default="data/tasks.jsonl", help="Path to tasks.jsonl")
    parser.add_argument("--data-dir", default="data", help="Root data directory")
    parser.add_argument("--task-id", help="Single instance_id to test")
    parser.add_argument("--repo", help="Filter tasks by repo (e.g. psf/requests, Textualize/rich)")
    parser.add_argument("--limit", type=int, default=5, help="Number of tasks to evaluate")
    parser.add_argument(
        "--mode",
        choices=["audit", "gold-eval"],
        default="gold-eval",
        help="Evaluation mode: audit (soundness) or gold-eval (evaluate reference fixes)"
    )
    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    data_dir = Path(args.data_dir)
    assert tasks_path.is_file(), f"Tasks file not found: {tasks_path}"

    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f]

    if args.task_id:
        tasks = [t for t in tasks if t["instance_id"] == args.task_id]
    if args.repo:
        tasks = [t for t in tasks if t["repo"] == args.repo]

    tasks = tasks[:args.limit]
    print(f"Loaded {len(tasks)} tasks for evaluation mode '{args.mode}'.")

    temp_root = Path(tempfile.mkdtemp(prefix="swe_bench_eval_"))
    results = []

    try:
        for idx, task in enumerate(tasks, 1):
            iid = task["instance_id"]
            repo = task["repo"]
            print(f"[{idx}/{len(tasks)}] Evaluating {iid} ({repo})...", end="", flush=True)

            if args.mode == "audit":
                res = evaluate_task_soundness(task, data_dir, temp_root)
            else:
                res = evaluate_patch(task, task["patch"], data_dir, temp_root)

            print(f" -> {res['status']}")
            results.append(res)

        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        if args.mode == "gold-eval":
            resolved = sum(1 for r in results if r.get("resolved"))
            total = len(results)
            print(f"Resolution Rate: {resolved}/{total} ({resolved/total*100:.1f}%)")
        else:
            sound = sum(1 for r in results if r.get("sound"))
            total = len(results)
            print(f"Sound Tasks: {sound}/{total} ({sound/total*100:.1f}%)")

        out_json = Path("evaluation_results.json")
        out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Detailed results written to: {out_json.resolve()}")

    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

if __name__ == "__main__":
    main()
