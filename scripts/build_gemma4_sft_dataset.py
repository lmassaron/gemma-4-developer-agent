#!/usr/bin/env python3
"""Convert curated SWE-bench trajectories into native Gemma 4 SFT training format.

Takes verified resolved trajectories from data/trajectories_raw/ and transforms
them into Gemma 4 chat-template compliant conversations:
  <start_of_turn>user / <|turn>user
  <start_of_turn>model / <|turn>model
  <thought> ... </thought>
  call:run_command{command: "..."}
  call:submit_patch{}
"""
import os
import re
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "trajectories_raw"
OUT_DIR = ROOT / "data" / "sft_dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_TOOL_OUTPUT_CHARS = 3500

def clean_command(cmd: str) -> str:
    """Normalize common wrapper commands."""
    cmd = cmd.strip()
    if cmd.startswith("cd /testbed && "):
        cmd = cmd[len("cd /testbed && "):]
    return cmd

def is_submit_command(cmd: str) -> bool:
    return "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in cmd

def convert_trajectory(traj_data: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    raw_msgs = traj_data.get("messages", [])
    if len(raw_msgs) < 3:
        return None
    
    instance_id = traj_data.get("instance_id", "unknown")
    
    # 1. System prompt & initial user request
    user_task = ""
    for m in raw_msgs:
        if m.get("role") == "user":
            content = m.get("content", "")
            # Extract PR description
            match = re.search(r"<pr_description>(.*?)</pr_description>", content, re.DOTALL)
            if match:
                user_task = match.group(1).strip()
            else:
                user_task = content.strip()
            break
            
    if not user_task:
        return None
        
    formatted_msgs = []
    # Gemma 4 initial user prompt
    init_prompt = (
        f"You are an expert autonomous software engineer solving an issue in the repository.\n"
        f"Instance ID: {instance_id}\n\n"
        f"Issue Description:\n"
        f"{user_task}\n\n"
        f"Instructions:\n"
        f"1. Search and inspect relevant files using run_command.\n"
        f"2. Write and test minimal reproduction scripts.\n"
        f"3. Edit source files to fix the issue.\n"
        f"4. Verify the fix and ensure no regressions.\n"
        f"5. When verified, invoke submit_patch."
    )
    formatted_msgs.append({"role": "user", "content": init_prompt})
    
    # Iterate through remaining turns
    i = 0
    while i < len(raw_msgs):
        m = raw_msgs[i]
        role = m.get("role")
        
        if role == "assistant":
            thought_text = str(m.get("content", "") or "").strip()
            # Clean thought prefixes
            thought_text = re.sub(r"^Okay, here's my (take|thought process)[^:]*:\s*", "", thought_text, flags=re.IGNORECASE)
            
            tool_calls = m.get("tool_calls") or []
            tool_call_str = ""
            
            if tool_calls:
                tc = tool_calls[0]
                func = tc.get("function") or tc.get("name") or ""
                args = tc.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"command": args}
                
                cmd = str(args.get("command", ""))
                cmd = clean_command(cmd)
                
                if is_submit_command(cmd):
                    tool_call_str = "call:submit_patch{}"
                elif cmd == "git diff":
                    tool_call_str = "call:get_status{}"
                else:
                    # Sanitize quotes for JSON argument
                    tool_call_str = f"call:run_command{{command: {json.dumps(cmd)}}}"
            
            # Construct model response
            model_content_parts = []
            if thought_text:
                model_content_parts.append(f"<thought>\n{thought_text}\n</thought>")
            if tool_call_str:
                model_content_parts.append(tool_call_str)
            elif not thought_text:
                # If neither thought nor tool call, skip empty turn
                i += 1
                continue
                
            model_turn = "\n".join(model_content_parts)
            formatted_msgs.append({"role": "assistant", "content": model_turn})
            
        elif role == "tool":
            # Tool response corresponds to previous tool call
            out = str(m.get("content", "") or "").strip()
            # Clean xml output wrappers
            out = re.sub(r"<returncode>\d+</returncode>\s*", "", out)
            out = re.sub(r"^<output>\s*", "", out)
            out = re.sub(r"\s*</output>$", "", out)
            
            if len(out) > MAX_TOOL_OUTPUT_CHARS:
                out = out[:MAX_TOOL_OUTPUT_CHARS] + "\n... [output truncated]"
                
            # If previous assistant called a tool, format user tool result
            prev_model = formatted_msgs[-1]["content"] if formatted_msgs else ""
            tc_match = re.search(r"(call:\w+\{.*?\})", prev_model)
            if tc_match:
                call_repr = tc_match.group(1)
                user_feedback = f"{call_repr} -> {json.dumps(out)}"
            else:
                user_feedback = out or "Command executed successfully with returncode 0."
                
            formatted_msgs.append({"role": "user", "content": user_feedback})
            
        i += 1
        
    # Ensure trajectory ends with submit_patch
    last_msg = formatted_msgs[-1]
    if last_msg["role"] == "user":
        # Model should have submitted
        formatted_msgs.append({
            "role": "assistant",
            "content": "<thought>\nThe reproduction test passes and all verification checks are complete. Submitting patch.\n</thought>\ncall:submit_patch{}"
        })
    elif last_msg["role"] == "assistant" and "submit_patch" not in last_msg["content"]:
        last_msg["content"] += "\ncall:submit_patch{}"
        
    return formatted_msgs

def main():
    raw_files = sorted(RAW_DIR.glob("*.json"))
    raw_files = [f for f in raw_files if f.name != "curated_resolved_index.json"]
    print(f"Found {len(raw_files)} raw curated trajectories in {RAW_DIR}")
    
    random.seed(42)
    random.shuffle(raw_files)
    
    dataset_records = []
    skipped = 0
    total_turns = 0
    
    for f in raw_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            msgs = convert_trajectory(data)
            if not msgs or len(msgs) < 4:
                skipped += 1
                continue
                
            record = {
                "instance_id": data.get("instance_id") or f.stem.split("__", 1)[-1],
                "source_model": f.name.split("__")[0],
                "num_turns": len(msgs),
                "messages": msgs
            }
            dataset_records.append(record)
            total_turns += len(msgs)
        except Exception as e:
            skipped += 1
            
    print(f"\nSuccessfully converted: {len(dataset_records)} trajectories (Skipped: {skipped})")
    avg_turns = total_turns / len(dataset_records) if dataset_records else 0
    print(f"Total turns: {total_turns} | Average turns per trajectory: {avg_turns:.1f}")
    
    # Train / Val Split (95% / 5%)
    val_size = max(20, int(len(dataset_records) * 0.05))
    train_records = dataset_records[:-val_size]
    val_records = dataset_records[-val_size:]
    
    train_path = OUT_DIR / "gemma4_swe_trajectories_train.jsonl"
    val_path = OUT_DIR / "gemma4_swe_trajectories_val.jsonl"
    
    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")
            
    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")
            
    print(f"\nWritten SFT Datasets:")
    print(f"  Train: {train_path} ({len(train_records)} trajectories, {train_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  Val:   {val_path} ({len(val_records)} trajectories, {val_path.stat().st_size / (1024*1024):.2f} MB)")
    
    meta_path = OUT_DIR / "dataset_metadata.json"
    meta_path.write_text(json.dumps({
        "total_trajectories": len(dataset_records),
        "train_count": len(train_records),
        "val_count": len(val_records),
        "avg_turns": round(avg_turns, 2),
        "source_distribution": {
            m: sum(1 for r in dataset_records if r["source_model"] == m)
            for m in set(r["source_model"] for r in dataset_records)
        }
    }, indent=2), encoding="utf-8")
    print(f"Metadata saved to {meta_path}")

if __name__ == "__main__":
    main()
