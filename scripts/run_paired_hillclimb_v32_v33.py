#!/home/lmassaron/code/venv_gemma4_12b/bin/python
"""
Paired Hill-Climbing Queue Runner for Track A (v32 Zero-Shot AP-BSSR) & Track B (v33 Native Causal LoRA AP-BSSR).

1. Evaluates Track A (`submission_v32_zero_shot_ap_bssr.zip`) on `core_12`.
2. Evaluates Track B (`submission_v33_native_causal_lora_ap_bssr.zip`) on `core_12`.
3. Computes the Paired McNemar Win/Loss matrix (`v32` vs `v33` vs `v30`) and promotes the winner
   to `submission.zip` and continues evaluation on `calibration_24` (24 tasks).
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/lmassaron/code/venv_gemma4_12b/bin/python"
EVAL_SCRIPT = ROOT / "scripts" / "evaluate_v21_finetuned_lora_full.py"
LOGS_DIR = ROOT / "state_of_the_art" / "benchmarks_data"
ARCHIVE_DIR = ROOT / "submissions_archive"


def run_eval(name: str, sub_zip: Path, suite: str, timeout_sec: int = 360, max_turns: int = 24) -> Path:
    label = f"{name}_{suite}"
    log_file = LOGS_DIR / f"benchmark_{label}.log"
    results_file = LOGS_DIR / f"benchmark_{label}_results.json"

    print(f"\n{'=' * 78}", flush=True)
    print(f" [HILL-CLIMB] STARTING: {name} on {suite} suite", flush=True)
    print(f" Submission : {sub_zip.name}", flush=True)
    print(f" Output JSON: {results_file.name}", flush=True)
    print(f" Log File   : {log_file.name}", flush=True)
    print(f"{'=' * 78}\n", flush=True)

    cmd = [
        PYTHON,
        str(EVAL_SCRIPT),
        "--suite", suite,
        "--label", label,
        "--submission_zip", str(sub_zip),
        "--timeout_sec", str(timeout_sec),
        "--max_turns", str(max_turns),
        "--output_file", str(results_file),
    ]

    with open(log_file, "a", encoding="utf-8") as f_out:
        f_out.write(f"\n--- [Hill-Climb] Starting {name} on {suite} at {time.ctime()} ---\n")
        f_out.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=f_out,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        proc.wait()

    print(f"[HILL-CLIMB] Finished {name} ({suite}) with return code {proc.returncode}", flush=True)
    return results_file


def compare_paired(res_a: Path, res_b: Path, res_v30: Path) -> dict:
    da = json.loads(res_a.read_text(encoding="utf-8")) if res_a.is_file() else {"tasks": []}
    db = json.loads(res_b.read_text(encoding="utf-8")) if res_b.is_file() else {"tasks": []}
    d30 = json.loads(res_v30.read_text(encoding="utf-8")) if res_v30.is_file() else {"tasks": []}

    map_a = {t["instance_id"]: t for t in da.get("tasks", [])}
    map_b = {t["instance_id"]: t for t in db.get("tasks", [])}
    map_30 = {t["instance_id"]: t for t in d30.get("tasks", [])}

    all_ids = sorted(set(map_a.keys()) | set(map_b.keys()))
    a_wins_vs_b, b_wins_vs_a = [], []
    a_wins_vs_30, b_wins_vs_30 = [], []

    print("\n" + "=" * 78)
    print(" PAIRED TASK-BY-TASK HILL-CLIMBING COMPARISON (Core-12)")
    print("=" * 78)
    print(f"{'Task ID':18s} | {'v30 (Old LoRA)':16s} | {'v32 (Zero-Shot)':16s} | {'v33 (Causal LoRA)':16s}")
    print("-" * 78)

    for tid in all_ids:
        r30 = map_30.get(tid, {})
        ra = map_a.get(tid, {})
        rb = map_b.get(tid, {})
        s30 = f"{'PASS' if r30.get('resolved') else r30.get('status', 'N/A')} ({r30.get('duration_sec', 0):.0f}s)"
        sa = f"{'PASS' if ra.get('resolved') else ra.get('status', 'N/A')} ({ra.get('duration_sec', 0):.0f}s)"
        sb = f"{'PASS' if rb.get('resolved') else rb.get('status', 'N/A')} ({rb.get('duration_sec', 0):.0f}s)"
        print(f"{tid:18s} | {s30:16s} | {sa:16s} | {sb:16s}")

        if ra.get("resolved") and not rb.get("resolved"):
            a_wins_vs_b.append(tid)
        elif rb.get("resolved") and not ra.get("resolved"):
            b_wins_vs_a.append(tid)

        if ra.get("resolved") and not r30.get("resolved"):
            a_wins_vs_30.append(tid)
        if rb.get("resolved") and not r30.get("resolved"):
            b_wins_vs_30.append(tid)

    res_count_a = sum(1 for t in map_a.values() if t.get("resolved"))
    res_count_b = sum(1 for t in map_b.values() if t.get("resolved"))
    dur_a = sum(t.get("duration_sec", 0) for t in map_a.values()) / max(1, len(map_a))
    dur_b = sum(t.get("duration_sec", 0) for t in map_b.values()) / max(1, len(map_b))

    summary = {
        "v32_resolved": res_count_a,
        "v33_resolved": res_count_b,
        "v32_mean_dur_sec": round(dur_a, 2),
        "v33_mean_dur_sec": round(dur_b, 2),
        "v32_wins_over_v33": a_wins_vs_b,
        "v33_wins_over_v32": b_wins_vs_a,
        "v32_wins_over_v30": a_wins_vs_30,
        "v33_wins_over_v30": b_wins_vs_30,
    }
    print("-" * 78)
    print(f"Summary: v32={res_count_a}/{len(all_ids)} ({dur_a:.1f}s) | v33={res_count_b}/{len(all_ids)} ({dur_b:.1f}s)")
    print(f"Discordant Wins: v32 over v33={a_wins_vs_b} | v33 over v32={b_wins_vs_a}")
    return summary


def main():
    v32_zip = ARCHIVE_DIR / "submission_v32_zero_shot_ap_bssr.zip"
    v33_zip = ARCHIVE_DIR / "submission_v33_native_causal_lora_ap_bssr.zip"
    v30_res = LOGS_DIR / "benchmark_v30_tuned_attn_lora_core12_results.json"

    res_v32 = run_eval("v32_zero_shot_ap_bssr", v32_zip, "core_12", timeout_sec=480, max_turns=28)
    res_v33 = run_eval("v33_native_causal_lora_ap_bssr", v33_zip, "core_12", timeout_sec=480, max_turns=28)

    summary = compare_paired(res_v32, res_v33, v30_res)
    out_summary = LOGS_DIR / "paired_hillclimb_v32_vs_v33_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Select winner (tie-breaker: higher resolved, then lower duration, or v33 if equal & NLL gate passed)
    if summary["v33_resolved"] > summary["v32_resolved"] or (
        summary["v33_resolved"] == summary["v32_resolved"] and summary["v33_mean_dur_sec"] <= summary["v32_mean_dur_sec"]
    ):
        winner_zip = v33_zip
        winner_name = "v33_native_causal_lora_ap_bssr"
    else:
        winner_zip = v32_zip
        winner_name = "v32_zero_shot_ap_bssr"

    shutil.copy2(winner_zip, ROOT / "submission.zip")
    print(f"\n[WINNER PROMOTED] Copied {winner_zip.name} -> {ROOT / 'submission.zip'}", flush=True)

    # Continue hill-climbing winner on expanded calibration_24 suite (resuming from the 12 core_12 tasks!)
    cal24_res = LOGS_DIR / f"benchmark_{winner_name}_calibration_24_results.json"
    core12_res = res_v33 if winner_zip == v33_zip else res_v32
    if core12_res.is_file() and not cal24_res.is_file():
        shutil.copy2(core12_res, cal24_res)
    run_eval(winner_name, winner_zip, "calibration_24", timeout_sec=480, max_turns=28)


if __name__ == "__main__":
    main()

