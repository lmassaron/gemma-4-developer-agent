#!/usr/bin/env python3
"""
Test initializing PEFT LoRA on Gemma 4 31B on the NVIDIA DGX Spark.
Verifies:
1. Memory footprint of base model in bfloat16.
2. Attachment of LoRA adapter using language_model-scoped regex.
3. Verification of trainable parameters count (should be ~121.6M).
"""

import time
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
from peft import LoraConfig, get_peft_model

def main():
    model_id = "google/gemma-4-31B-it"
    print(f"Loading {model_id} in bfloat16 on DGX Spark...")
    t0 = time.time()
    
    base_model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    t1 = time.time()
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"Base model loaded in {t1 - t0:.2f}s.")
    print(f"VRAM Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")
    
    # Scoped target regex
    lora_target_regex = r".*language_model.*\.(q_proj|v_proj|k_proj|o_proj|gate_proj|up_proj|down_proj)"
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=lora_target_regex,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    print("Applying PEFT LoRA adapter...")
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    print("PEFT LoRA initialization on Gemma 4 31B: SUCCESS!")

if __name__ == "__main__":
    main()
