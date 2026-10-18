#!/usr/bin/env python3
"""
Trajectory-SFT: Supervised Fine-Tuning on Multi-Turn SWE Agent Trajectories
Optimized for Google Gemma 4 on NVIDIA DGX Spark (Grace Blackwell GB10 / CUDA 13.0).

Key features:
1. Turn-level teacher forcing with strict loss masking:
   - User issue prompt tokens: labels = -100
   - Tool execution outputs (<|tool_response> ... <tool_response|>): labels = -100
   - Assistant reasoning & tool calls (<|turn>model ... <turn|>): cross-entropy loss computed
2. PEFT LoRA adapter targeting language model attention and MLP projections:
   - q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj
3. Native bfloat16 mixed precision with PyTorch gradient checkpointing.
4. Auto-exports Hugging Face PEFT adapter for Kaggle submission bundling.
"""

import os
import sys
import json
import time
import math
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    AutoConfig,
    AutoModelForMultimodalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel


# Delimiter token constants for Gemma 4
TOKEN_TURN_START = 105      # <|turn>
TOKEN_USER = 2364           # user
TOKEN_MODEL = 4368          # model
TOKEN_TURN_END = 106        # <turn|>
TOKEN_TOOL_CALL_START = 48  # <|tool_call>
TOKEN_TOOL_CALL_END = 49    # <tool_call|>
TOKEN_TOOL_RESP_START = 50  # <|tool_response>
TOKEN_TOOL_RESP_END = 51    # <tool_response|>


class SweTrajectoryDataset(Dataset):
    """
    Dataset that loads multi-turn developer trajectories and builds
    sub-window examples with exact assistant-turn loss masking.
    """
    def __init__(
        self,
        jsonl_path: str,
        processor: Any,
        max_seq_len: int = 2048,
        max_samples: Optional[int] = None,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_seq_len = max_seq_len
        self.examples = []

        print(f"[Dataset] Loading trajectories from {jsonl_path}...")
        start_time = time.time()
        raw_count = 0

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                raw_count += 1
                item = json.loads(line)

                # Each item has: instance_id, repo, issue, messages
                issue_text = item.get("issue", "")
                messages = item.get("messages", [])

                # We break trajectories into sub-windows ending on assistant turns
                conv = [{"role": "user", "content": issue_text}]
                for m in messages:
                    role = m.get("role")
                    if role == "assistant":
                        conv.append({
                            "role": "assistant",
                            "content": m.get("reasoning") or "",
                            "tool_calls": m.get("tool_calls")
                        })
                        # Add snapshot at assistant turn
                        if len(conv) >= 2:
                            self.examples.append(list(conv))
                    elif role == "tool":
                        conv.append({
                            "role": "tool",
                            "tool_call_id": m.get("tool_call_id", "call_0"),
                            "name": m.get("name", "run_command"),
                            "content": str(m.get("content", ""))
                        })

                if max_samples and len(self.examples) >= max_samples:
                    break

        print(f"[Dataset] Processed {raw_count} trajectories into {len(self.examples)} turn-windows in {time.time() - start_time:.2f}s")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        conv = self.examples[idx]
        try:
            formatted = self.processor.apply_chat_template(conv, tokenize=False)
        except Exception:
            # Fallback simple text formatting if jinja fails on edge cases
            formatted = f"<bos><|turn>user\n{conv[0]['content']}<turn|>\n<|turn>model\n{conv[-1].get('content', '')}<turn|>"

        token_ids = self.tokenizer.encode(formatted, add_special_tokens=False)

        # Truncate from beginning if exceeding max_seq_len (preserve latest context & target)
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[-self.max_seq_len:]

        # Create labels with selective loss masking (-100 on user & tool observations)
        labels = [-100] * len(token_ids)
        in_model = False
        in_tool_call = False
        in_tool_resp = False

        for i in range(len(token_ids)):
            t = token_ids[i]
            if t == TOKEN_TURN_START:
                if i + 1 < len(token_ids) and token_ids[i + 1] == TOKEN_MODEL:
                    in_model = True
                else:
                    in_model = False
            elif t == TOKEN_TURN_END:
                if in_model or in_tool_call:
                    labels[i] = t
                    in_model = False
                    in_tool_call = False
                continue
            elif t == TOKEN_TOOL_CALL_START:
                in_tool_call = True
            elif t == TOKEN_TOOL_CALL_END:
                labels[i] = t
                in_tool_call = False
                continue
            elif t == TOKEN_TOOL_RESP_START:
                in_tool_resp = True
            elif t == TOKEN_TOOL_RESP_END:
                in_tool_resp = False
                continue

            if (in_model and not in_tool_resp) or in_tool_call:
                labels[i] = t

        # Safety fallback: ensure at least the final tokens are learned if all -100
        labels_t = torch.tensor(labels, dtype=torch.long)
        if (labels_t != -100).sum() == 0:
            labels_t[-1] = token_ids[-1]

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "labels": labels_t,
            "attention_mask": torch.ones(len(token_ids), dtype=torch.long),
            "mm_token_type_ids": torch.zeros(len(token_ids), dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad batch dynamically to longest sequence in batch."""
    max_len = max(len(x["input_ids"]) for x in batch)
    pad_token_id = 0

    batch_input_ids = []
    batch_labels = []
    batch_attention_mask = []
    batch_mm_token_type_ids = []

    for item in batch:
        cur_len = len(item["input_ids"])
        pad_len = max_len - cur_len

        # Pad with 0 for inputs, -100 for labels, 0 for attention and mm_token_type_ids
        padded_inputs = torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        padded_labels = torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        padded_mask = torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        padded_mm = torch.cat([item["mm_token_type_ids"], torch.zeros(pad_len, dtype=torch.long)])

        batch_input_ids.append(padded_inputs)
        batch_labels.append(padded_labels)
        batch_attention_mask.append(padded_mask)
        batch_mm_token_type_ids.append(padded_mm)

    return {
        "input_ids": torch.stack(batch_input_ids),
        "labels": torch.stack(batch_labels),
        "attention_mask": torch.stack(batch_attention_mask),
        "mm_token_type_ids": torch.stack(batch_mm_token_type_ids),
    }


def main():
    parser = argparse.ArgumentParser(description="Gemma 4 Trajectory SFT Fine-Tuning")
    parser.add_argument("--model_id", type=str, default="google/gemma-4-E2B-it", help="Base model ID or path")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to conv_curated_v2.jsonl")
    parser.add_argument("--output_dir", type=str, default="adapters/gemma4_swe_lora", help="Output directory for LoRA adapter")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Max sequence length per window")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=100, help="Max training steps (0 for full dataset)")
    parser.add_argument("--save_steps", type=int, default=25, help="Save checkpoint every N steps")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank r")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    args = parser.parse_args()

    # System verification
    print("=" * 65)
    print("      GEMMA 4 TRAJECTORY-SFT FINE-TUNING PIPELINE")
    print("=" * 65)
    print(f"Base Model        : {args.model_id}")
    print(f"Dataset           : {args.dataset_path}")
    print(f"Output Adapter Dir: {args.output_dir}")
    print(f"Sequence Length   : {args.max_seq_len} tokens")
    print(f"Effective Batch   : {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"LoRA Rank / Alpha : r={args.lora_r}, alpha={args.lora_alpha}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        props = torch.cuda.get_device_properties(0)
        print(f"Hardware          : {props.name} ({props.total_memory / 1e9:.2f} GB VRAM)")
    else:
        print("Hardware          : CPU (Warning: CUDA strongly recommended)")
    print("=" * 65)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Processor
    print("\n[1/5] Loading AutoProcessor and Tokenizer...")
    processor = AutoProcessor.from_pretrained(args.model_id)

    # 2. Build Dataset
    print("\n[2/5] Building Trajectory Dataset...")
    dataset = SweTrajectoryDataset(
        jsonl_path=args.dataset_path,
        processor=processor,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_steps * args.batch_size * args.grad_accum if args.max_steps > 0 else None,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # 3. Load Base Model
    print(f"\n[3/5] Loading Base Model ({args.model_id}) with bfloat16...")
    t_load = time.time()
    base_model = AutoModelForMultimodalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Base model loaded in {time.time() - t_load:.2f}s. GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # 4. Attach PEFT LoRA Adapter
    print("\n[4/5] Initializing PEFT LoRA Adapter...")
    # Target only the text language model attention and MLP projections
    lora_target_regex = r".*language_model.*\.(q_proj|v_proj|k_proj|o_proj|gate_proj|up_proj|down_proj)"
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=lora_target_regex,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # 5. Training Loop
    total_steps = args.max_steps if args.max_steps > 0 else (len(dataloader) // args.grad_accum) * args.epochs
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    warmup_steps = max(5, int(0.05 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    print(f"\n[5/5] Starting Trajectory-SFT Training ({total_steps} optimizer steps)...")
    model.train()
    micro_step = 0
    step = 0
    accum_loss = 0.0
    start_train_time = time.time()

    data_iter = iter(dataloader)
    optimizer.zero_grad()

    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        mm_token_type_ids = batch["mm_token_type_ids"].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
        )
        if torch.isnan(outputs.loss):
            print(f"[Warning] NaN loss encountered at micro_step {micro_step}, skipping...")
            optimizer.zero_grad()
            continue

        loss = outputs.loss / args.grad_accum
        loss.backward()
        accum_loss += loss.item()
        micro_step += 1

        if micro_step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            current_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_train_time
            steps_per_sec = step / elapsed if elapsed > 0 else 0
            vram_gb = torch.cuda.memory_allocated() / 1e9

            print(f"Step {step:03d}/{total_steps:03d} | Loss: {accum_loss:.4f} | LR: {current_lr:.2e} | VRAM: {vram_gb:.2f} GB | Speed: {steps_per_sec:.2f} step/s")
            accum_loss = 0.0

            if step % args.save_steps == 0 or step == total_steps:
                checkpoint_dir = Path(args.output_dir) / f"checkpoint-{step}"
                print(f"--> Saving checkpoint to {checkpoint_dir}...")
                model.save_pretrained(checkpoint_dir)
                processor.save_pretrained(checkpoint_dir)

    # Final save
    print(f"\nTraining completed in {time.time() - start_train_time:.2f}s!")
    print(f"Saving final LoRA adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # Verification: test adapter reloading
    print("\n[Verification] Reloading trained LoRA adapter...")
    test_adapter = PeftModel.from_pretrained(base_model, args.output_dir)
    print("SUCCESS: Adapter verified and fully loadable!")
    print(f"Artifacts stored in: {args.output_dir}")


if __name__ == "__main__":
    main()
