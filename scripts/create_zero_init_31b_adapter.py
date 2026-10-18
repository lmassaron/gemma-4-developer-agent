#!/usr/bin/env python3
"""
Create a zero-initialized identity LoRA adapter for Google Gemma 4 31B.

Dimensions for Gemma 4 31B:
- hidden_size: 5376
- num_hidden_layers: 60
- num_attention_heads: 32 (head_dim: 256 -> q_dim: 8192)
- num_key_value_heads: 16 (head_dim: 256 -> kv_dim: 4096)
- intermediate_size: 21504
- LoRA rank r = 16, alpha = 32

Because all B matrices are initialized to exact zeros, delta_W = B * A = 0.
This acts as a pure mathematical identity transformation to test whether
the Kaggle / vLLM serving harness successfully loads LoRA adapters over W4A16.
"""

import json
from pathlib import Path
import torch
from safetensors.torch import save_file

def build_zero_adapter(output_dir: Path, rank: int = 16, alpha: int = 32):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    hidden_size = 5376
    q_dim = 32 * 256     # 8192
    kv_dim = 16 * 256    # 4096
    inter_dim = 21504
    num_layers = 60
    
    tensors = {}
    
    for layer in range(num_layers):
        prefix = f"base_model.model.model.language_model.layers.{layer}"
        
        # 1. self_attn.q_proj (in: 5376, out: 8192)
        tensors[f"{prefix}.self_attn.q_proj.lora_A.weight"] = torch.randn(rank, hidden_size, dtype=torch.bfloat16) * 0.01
        tensors[f"{prefix}.self_attn.q_proj.lora_B.weight"] = torch.zeros(q_dim, rank, dtype=torch.bfloat16)
        
        # 2. self_attn.k_proj (in: 5376, out: 4096)
        tensors[f"{prefix}.self_attn.k_proj.lora_A.weight"] = torch.randn(rank, hidden_size, dtype=torch.bfloat16) * 0.01
        tensors[f"{prefix}.self_attn.k_proj.lora_B.weight"] = torch.zeros(kv_dim, rank, dtype=torch.bfloat16)
        
        # 3. self_attn.v_proj (in: 5376, out: 4096)
        tensors[f"{prefix}.self_attn.v_proj.lora_A.weight"] = torch.randn(rank, hidden_size, dtype=torch.bfloat16) * 0.01
        tensors[f"{prefix}.self_attn.v_proj.lora_B.weight"] = torch.zeros(kv_dim, rank, dtype=torch.bfloat16)
        
        # 4. self_attn.o_proj (in: 8192, out: 5376)
        tensors[f"{prefix}.self_attn.o_proj.lora_A.weight"] = torch.randn(rank, q_dim, dtype=torch.bfloat16) * 0.01
        tensors[f"{prefix}.self_attn.o_proj.lora_B.weight"] = torch.zeros(hidden_size, rank, dtype=torch.bfloat16)
        
        # 5. mlp.gate_proj (in: 5376, out: 21504)
        tensors[f"{prefix}.mlp.gate_proj.lora_A.weight"] = torch.randn(rank, hidden_size, dtype=torch.bfloat16) * 0.01
        tensors[f"{prefix}.mlp.gate_proj.lora_B.weight"] = torch.zeros(inter_dim, rank, dtype=torch.bfloat16)
        
        # 6. mlp.up_proj (in: 5376, out: 21504)
        tensors[f"{prefix}.mlp.up_proj.lora_A.weight"] = torch.randn(rank, hidden_size, dtype=torch.bfloat16) * 0.01
        tensors[f"{prefix}.mlp.up_proj.lora_B.weight"] = torch.zeros(inter_dim, rank, dtype=torch.bfloat16)
        
        # 7. mlp.down_proj (in: 21504, out: 5376)
        tensors[f"{prefix}.mlp.down_proj.lora_A.weight"] = torch.randn(rank, inter_dim, dtype=torch.bfloat16) * 0.01
        tensors[f"{prefix}.mlp.down_proj.lora_B.weight"] = torch.zeros(hidden_size, rank, dtype=torch.bfloat16)

    # Save safetensors
    weights_path = output_dir / "adapter_model.safetensors"
    save_file(tensors, weights_path)
    
    # Save adapter_config.json
    config = {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": "google/gemma-4-31B-it",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": alpha,
        "lora_bias": False,
        "lora_dropout": 0.05,
        "modules_to_save": None,
        "peft_type": "LORA",
        "peft_version": "0.21.2",
        "r": rank,
        "rank_pattern": {},
        "revision": None,
        "target_modules": ".*language_model.*\\.(q_proj|v_proj|k_proj|o_proj|gate_proj|up_proj|down_proj)",
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False
    }
    
    config_path = output_dir / "adapter_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    
    print(f"Successfully generated zero-init 31B identity adapter at: {output_dir}")
    print(f"  - Layers: {num_layers}")
    print(f"  - Tensor keys: {len(tensors)} (60 layers * 7 modules * 2 matrices = 840 tensors)")
    print(f"  - File size: {weights_path.stat().st_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="adapters/gemma4_31b_identity_test")
    args = parser.parse_args()
    build_zero_adapter(Path(args.output))
