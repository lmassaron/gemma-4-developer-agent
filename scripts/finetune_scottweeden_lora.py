#!/usr/bin/env python3
"""Continuing fine-tuning of Scott Weeden's rank-16 LoRA on Gemma 4 31B QAT.

Uses the 2,250 verified resolved SWE-bench gold trajectories in data/sft_dataset/.
Trains with selective loss masking (loss only on model reasoning and tool call tokens).
Saves the fine-tuned adapter to models/lora_v21_scottweeden_finetuned.
"""
import os
import sys
import json
import time
import math
import random
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel

# Monkey-patches for compressed-tensors + Gemma4ClippableLinear + PEFT
import peft.tuners.lora.layer as lora_layer_module
import peft.tuners.lora.model as lora_model_module
import peft.tuners.tuners_utils as tu
from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear

orig_tu_in_out = tu._get_in_out_features
def custom_tu_in_out(module):
    if hasattr(module, "in_features") and hasattr(module, "out_features"):
        return module.in_features, module.out_features
    if hasattr(module, "linear") and hasattr(module.linear, "in_features"):
        return module.linear.in_features, module.linear.out_features
    return orig_tu_in_out(module)
tu._get_in_out_features = custom_tu_in_out

def custom_layer_in_out(self, root_module):
    if hasattr(root_module, "in_features") and hasattr(root_module, "out_features"):
        return root_module.in_features, root_module.out_features
    if hasattr(root_module, "linear") and hasattr(root_module.linear, "in_features"):
        return root_module.linear.in_features, root_module.linear.out_features
    return orig_tu_in_out(root_module)
lora_layer_module.Linear._get_in_out_features = custom_layer_in_out

orig_create_new_module = lora_model_module.LoraModel._create_new_module
@staticmethod
def custom_create_new_module(lora_config, adapter_name, target, **kwargs):
    if isinstance(target, Gemma4ClippableLinear):
        return orig_create_new_module(lora_config, adapter_name, target.linear, **kwargs)
    return orig_create_new_module(lora_config, adapter_name, target, **kwargs)
lora_model_module.LoraModel._create_new_module = custom_create_new_module

BASE_MODEL_PATH = "/home/lmassaron/models/kaggle/gemma-4-31b-it-qat-w4a16-ct"
INITIAL_LORA_PATH = "/home/lmassaron/models/kaggle/lora_scottweeden"
OUTPUT_LORA_PATH = ROOT / "models" / "lora_v21_scottweeden_finetuned"
DATA_FILE = ROOT / "data" / "sft_dataset" / "gemma4_swe_trajectories_train.jsonl"
VAL_FILE = ROOT / "data" / "sft_dataset" / "gemma4_swe_trajectories_val.jsonl"

MAX_SEQ_LEN = 3072

class SWETrajectoryDataset(Dataset):
    def __init__(self, jsonl_path: Path, processor: AutoProcessor, max_samples: int = 500):
        self.samples = []
        self.processor = processor
        
        print(f"Loading data from {jsonl_path} (max_samples={max_samples})...")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                self.samples.append(d["messages"])
                if len(self.samples) >= max_samples:
                    break
        print(f"Loaded {len(self.samples)} trajectory conversations.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        messages = self.samples[idx]
        
        # Tokenize full conversation with loss masking on user turns
        input_ids = []
        labels = []
        
        # Add BOS
        bos_id = self.processor.tokenizer.bos_token_id
        if bos_id is not None:
            input_ids.append(bos_id)
            labels.append(-100)
            
        for m in messages:
            role = m["role"]
            content = m["content"]
            
            # Format turn
            turn_str = f"<|turn>{'user' if role == 'user' else 'model'}\n{content}<turn|>\n"
            turn_tokens = self.processor.tokenizer.encode(turn_str, add_special_tokens=False)
            
            if len(input_ids) + len(turn_tokens) > MAX_SEQ_LEN:
                break
                
            input_ids.extend(turn_tokens)
            if role == "assistant":
                # Supervised loss on assistant tokens
                labels.extend(turn_tokens)
            else:
                # Mask user tokens
                labels.extend([-100] * len(turn_tokens))
                
        # Ensure at least one assistant token is supervised
        if all(lbl == -100 for lbl in labels):
            labels[-1] = input_ids[-1]
            
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def collate_fn(batch, pad_token_id=0):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids_padded = []
    labels_padded = []
    attention_mask = []
    
    for b in batch:
        seq_len = len(b["input_ids"])
        pad_len = max_len - seq_len
        
        inp = torch.cat([b["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        lbl = torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        mask = torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
        
        input_ids_padded.append(inp)
        labels_padded.append(lbl)
        attention_mask.append(mask)
        
    return {
        "input_ids": torch.stack(input_ids_padded),
        "labels": torch.stack(labels_padded),
        "attention_mask": torch.stack(attention_mask)
    }

def main():
    device = "cuda:0"
    print("=" * 60)
    print(" STARTING LORA CONTINUING FINE-TUNING")
    print(f" Base: {BASE_MODEL_PATH}")
    print(f" Initial Adapter: {INITIAL_LORA_PATH}")
    print(f" Target Output: {OUTPUT_LORA_PATH}")
    print("=" * 60)
    
    # 1. Load Processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)
    pad_id = processor.tokenizer.pad_token_id or 0
    
    # 2. Build Dataset
    train_dataset = SWETrajectoryDataset(DATA_FILE, processor, max_samples=400)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_id)
    )
    
    # 3. Load Model
    print("Loading QAT base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype="auto",
        device_map=device
    )
    
    print(f"Loading initial LoRA from {INITIAL_LORA_PATH} (is_trainable=True)...")
    model = PeftModel.from_pretrained(model, INITIAL_LORA_PATH, is_trainable=True)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"LoRA Attached! Trainable parameters: {num_trainable:,}")
    
    # 4. Optimizer & Scheduler
    # Fine-tuning rate: 5e-5 for 400 high-quality trajectories with grad accumulation 8
    lr = 5e-5
    grad_accum_steps = 8
    num_epochs = 1
    total_steps = (len(train_loader) // grad_accum_steps) * num_epochs
    
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    
    print(f"Training setup:")
    print(f"  Total samples: {len(train_dataset)}")
    print(f"  Gradient accumulation steps: {grad_accum_steps}")
    print(f"  Effective batch size: {grad_accum_steps}")
    print(f"  Total optimization steps: {total_steps}")
    print(f"  Learning rate: {lr}")
    
    # 5. Training Loop
    model.train()
    step = 0
    running_loss = 0.0
    start_time = time.time()
    
    optimizer.zero_grad()
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        for batch_idx, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            running_loss += loss.item() * grad_accum_steps
            
            if batch_idx % grad_accum_steps == 0 or batch_idx == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                
                avg_loss = running_loss / (grad_accum_steps if batch_idx % grad_accum_steps == 0 else (batch_idx % grad_accum_steps))
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed if elapsed > 0 else 0
                eta = (total_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
                
                if step % 5 == 0 or step == total_steps:
                    print(f"Step [{step}/{total_steps}] | Loss: {avg_loss:.4f} | "
                          f"Elapsed: {elapsed:.1f}s | Speed: {steps_per_sec:.2f} step/s | ETA: {eta:.0f}s")
                running_loss = 0.0
                
    total_train_time = round(time.time() - start_time, 1)
    print(f"\nTraining completed in {total_train_time}s!")
    
    # 6. Save Fine-Tuned Adapter
    OUTPUT_LORA_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Saving fine-tuned LoRA adapter to {OUTPUT_LORA_PATH}...")
    model.save_pretrained(str(OUTPUT_LORA_PATH))
    print("Fine-tuned LoRA saved successfully!")
    
    # Verify saved files
    saved_files = list(OUTPUT_LORA_PATH.glob("*"))
    print("Saved files:", [f.name for f in saved_files])

if __name__ == "__main__":
    main()
