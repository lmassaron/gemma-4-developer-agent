#!/usr/bin/env python3
"""Inspect and validate the converted Gemma 4 SWE SFT dataset."""
import json
import re
from pathlib import Path
from collections import Counter
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "sft_dataset"
TRAIN_FILE = DATA_DIR / "gemma4_swe_trajectories_train.jsonl"
BASE_MODEL_PATH = "/home/lmassaron/models/kaggle/gemma-4-31b-it-qat-w4a16-ct"

def main():
    print(f"Loading Gemma 4 processor from {BASE_MODEL_PATH}...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)
    
    print(f"Reading dataset: {TRAIN_FILE}...")
    num_samples = 0
    tool_counter = Counter()
    thought_lens = []
    
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            num_samples += 1
            data = json.loads(line)
            msgs = data["messages"]
            
            for m in msgs:
                if m["role"] == "assistant":
                    content = m["content"]
                    # Count tool calls
                    for tc in re.findall(r"call:(\w+)\{", content):
                        tool_counter[tc] += 1
                    # Extract thoughts
                    t_match = re.search(r"<thought>(.*?)</thought>", content, re.DOTALL)
                    if t_match:
                        thought_lens.append(len(t_match.group(1).split()))
                        
            if idx == 0:
                print("\n==========================================")
                print(f" SAMPLE TRAJECTORY: {data['instance_id']} ({data['source_model']})")
                print("==========================================")
                # Show first 4 turns
                for m_idx, m in enumerate(msgs[:4]):
                    print(f"\n--- Turn {m_idx} ({m['role']}) ---")
                    c = m['content']
                    if len(c) > 300:
                        print(c[:300] + "\n... [truncated]")
                    else:
                        print(c)
                print("\n... [continues for", len(msgs), "total turns] ...\n")
                
                # Check template rendering
                rendered = processor.apply_chat_template(msgs[:6], tokenize=False)
                print("==========================================")
                print(" RENDERED GEMMA 4 CHAT TEMPLATE (Turns 0-5):")
                print("==========================================")
                print(rendered[:500] + "\n...")

    print("\n==========================================")
    print(" DATASET SUMMARY METRICS")
    print("==========================================")
    print(f"Total Trajectories: {num_samples}")
    print(f"Total Tool Invocations Across Dataset:")
    for t_name, cnt in tool_counter.most_common():
        print(f"  - call:{t_name}: {cnt:,} calls")
    if thought_lens:
        avg_thought_words = sum(thought_lens) / len(thought_lens)
        print(f"Total Thought Blocks: {len(thought_lens):,}")
        print(f"Average Words per Thought Block: {avg_thought_words:.1f} words")

if __name__ == "__main__":
    main()
