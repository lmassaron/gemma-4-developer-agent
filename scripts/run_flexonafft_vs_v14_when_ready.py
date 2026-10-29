#!/usr/bin/env python3
"""
Automated Watcher & GPU Transition Script.
Monitors PID 3425992 (Candidate 5 on 60-task benchmark).
Once PID 3425992 finishes and releases GPU VRAM:
Automatically starts benchmark_flexonafft_vs_v14.py with real-time logging.
"""

import os
import subprocess
import time
import sys
from pathlib import Path

TARGET_PID = 3425992
LOG_DIR = Path("state_of_the_art/benchmarks_data")
LOG_DIR.mkdir(parents=True, exist_ok=True)
WATCH_LOG = LOG_DIR / "watcher_gpu_queue.log"
BENCH_LOG = LOG_DIR / "bench_flexonafft_vs_v14.log"

def is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def get_gpu_memory_used() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        ).decode().strip()
        return int(out.split("\n")[0])
    except Exception:
        return 0

def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    with open(WATCH_LOG, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")

def main():
    log(f"Starting GPU Transition Watcher. Monitoring PID {TARGET_PID}...")
    
    poll_count = 0
    while is_pid_running(TARGET_PID):
        poll_count += 1
        if poll_count % 12 == 0:  # Every 60s
            log(f"PID {TARGET_PID} is still running. GPU memory in use: {get_gpu_memory_used()} MiB.")
        time.sleep(5)

    log(f"PID {TARGET_PID} has completed! Waiting for GPU VRAM release...")
    
    # Wait for GPU memory to drop below 2000 MiB (idle)
    for _ in range(30):
        mem = get_gpu_memory_used()
        log(f"Current GPU Memory: {mem} MiB")
        if mem < 4000:
            log("GPU memory successfully released!")
            break
        time.sleep(2)

    log(f"Launching flexonafft_v2 vs v14 benchmark. Logging to {BENCH_LOG}...")
    
    cmd = [
        sys.executable, "-u", "scripts/benchmark_flexonafft_vs_v14.py",
        "--candidate", "all",
        "--suite", "core_60", "--num_tasks", "60",
        "--output_file", "benchmark_flexonafft_vs_v14_results.json"
    ]
    
    with open(BENCH_LOG, "w", encoding="utf-8") as out_f:
        proc = subprocess.Popen(cmd, stdout=out_f, stderr=subprocess.STDOUT)
        log(f"Benchmark process started with PID {proc.pid}.")
        proc.wait()
        log(f"Benchmark process {proc.pid} finished with exit code {proc.returncode}.")

if __name__ == "__main__":
    main()
