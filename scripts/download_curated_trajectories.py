#!/usr/bin/env python3
"""Download and curate ground-truth resolved SWE-bench trajectories from Hugging Face.

Source: tarsur385/swebench-verified-trajectories
Models curated:
  - gemini-3-pro
  - gemini-3-flash-high
  - claude-4.5-opus-high
  - claude-opus-4.6
  - gpt-5.2-high
  - gpt-5.2-codex

Filters strictly for info.resolved == True / scores.resolved == 1.
"""
import os
import sys
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "trajectories_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TARGET_MODELS = [
    "gemini-3-pro",
    "gemini-3-flash-high",
    "claude-4.5-opus-high",
    "claude-opus-4.6",
    "gpt-5.2-high",
    "gpt-5.2-codex",
]

REPO_ID = "tarsur385/swebench-verified-trajectories"

def download_and_verify(rel_path: str):
    try:
        p = hf_hub_download(repo_id=REPO_ID, filename=rel_path, repo_type="dataset")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        info = data.get("info", {})
        is_resolved = bool(
            info.get("resolved") is True or 
            info.get("scores", {}).get("resolved") == 1
        )
        
        if not is_resolved:
            return None
        
        model_name = rel_path.split("/")[1]
        instance_id = data.get("instance_id") or Path(rel_path).stem.replace(".traj", "")
        
        dest_file = RAW_DIR / f"{model_name}__{instance_id}.json"
        with open(dest_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
        
        return {
            "model": model_name,
            "instance_id": instance_id,
            "num_messages": len(data.get("messages", [])),
            "cost": info.get("model_stats", {}).get("instance_cost", 0.0),
            "file": str(dest_file.name)
        }
    except Exception as e:
        print(f"Error downloading {rel_path}: {e}")
        return None

def main():
    print(f"Connecting to Hugging Face dataset: {REPO_ID}...")
    api = HfApi()
    all_files = api.list_repo_files(REPO_ID, repo_type="dataset")
    
    target_files = []
    for f in all_files:
        if not f.endswith(".traj.json"):
            continue
        parts = f.split("/")
        if len(parts) >= 3 and parts[1] in TARGET_MODELS:
            target_files.append(f)
            
    print(f"Found {len(target_files)} target trajectory files across {len(TARGET_MODELS)} models.")
    print(f"Target models: {TARGET_MODELS}")
    print(f"Starting parallel download and resolution verification (max_workers=20)...")
    
    start_time = time.time()
    curated = []
    processed = 0
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(download_and_verify, f): f for f in target_files}
        for fut in as_completed(futures):
            processed += 1
            res = fut.result()
            if res:
                curated.append(res)
            
            if processed % 100 == 0 or processed == len(target_files):
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"[{processed}/{len(target_files)}] Processed | "
                      f"Curated Resolved: {len(curated)} ({len(curated)/processed*100:.1f}%) | "
                      f"Speed: {rate:.1f} files/s")
                
    tot_time = time.time() - start_time
    print(f"\n==========================================")
    print(f" DOWNLOAD AND CURATION COMPLETE")
    print(f"==========================================")
    print(f"Total Trajectories Inspected: {processed}")
    print(f"Total Verified Resolved Trajectories: {len(curated)}")
    print(f"Elapsed Time: {tot_time:.1f}s")
    
    # Model breakdown
    breakdown = {}
    for c in curated:
        m = c["model"]
        breakdown[m] = breakdown.get(m, 0) + 1
        
    print("\nResolved Count by Model:")
    for m, count in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {m}: {count} resolved trajectories")
        
    index_file = RAW_DIR / "curated_resolved_index.json"
    index_file.write_text(json.dumps(curated, indent=2), encoding="utf-8")
    print(f"\nIndex saved to {index_file}")

if __name__ == "__main__":
    main()
