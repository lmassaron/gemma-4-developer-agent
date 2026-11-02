#!/usr/bin/env python3
"""
SVD Compression of LoRA Adapters for Google Gemma 4 Developer Agent.

Compresses a trained rank-R LoRA adapter (e.g. r=16) down to rank-K (e.g. r=4)
in closed form using Truncated Singular Value Decomposition (SVD) on the factorized
low-rank product B @ A.

No training data required. Mathematically optimal low-rank approximation under
the Eckart-Young-Mirsky theorem.
"""

import argparse
import json
from pathlib import Path
import shutil
import time
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]


def compress_lora(
    input_dir: Path,
    output_dir: Path,
    target_rank: int = 4,
    attention_only: bool = False,
    device: str = "cpu",
):
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_file = input_dir / "adapter_model.safetensors"
    config_file = input_dir / "adapter_config.json"

    assert weights_file.is_file(), f"Missing weights: {weights_file}"
    assert config_file.is_file(), f"Missing config: {config_file}"

    config = json.loads(config_file.read_text(encoding="utf-8"))
    source_rank = config.get("r", 16)
    source_alpha = config.get("lora_alpha", 32)
    scaling_ratio = source_alpha / source_rank

    assert target_rank < source_rank, (
        f"Target rank ({target_rank}) must be strictly smaller than source rank ({source_rank})"
    )

    print(f"Loading weights from {weights_file}...")
    t0 = time.time()
    state_dict = load_file(str(weights_file), device=device)
    print(f"Loaded {len(state_dict)} tensor keys in {time.time() - t0:.2f}s")

    # Find paired LoRA A and B keys
    lora_keys = {}
    for k in state_dict.keys():
        if "lora_A.weight" in k:
            prefix = k.replace("lora_A.weight", "")
            lora_keys.setdefault(prefix, {})["A"] = k
        elif "lora_B.weight" in k:
            prefix = k.replace("lora_B.weight", "")
            lora_keys.setdefault(prefix, {})["B"] = k

    print(f"Found {len(lora_keys)} paired LoRA modules to compress.")

    new_state_dict = {}
    target_modules = set()
    total_energy_ratio = []

    for idx, (prefix, pair) in enumerate(lora_keys.items(), 1):
        key_A = pair["A"]
        key_B = pair["B"]

        # Check attention-only filter
        is_attention = any(m in prefix for m in ["q_proj", "k_proj", "v_proj", "o_proj"])
        if attention_only and not is_attention:
            # Skip MLP modules
            continue

        weight_A = state_dict[key_A].float()  # [source_rank, in_dim]
        weight_B = state_dict[key_B].float()  # [out_dim, source_rank]
        orig_dtype = state_dict[key_A].dtype

        # 1. Compact QR factorizations
        QB, RB = torch.linalg.qr(weight_B)       # QB: [out_dim, r], RB: [r, r]
        QA, RA = torch.linalg.qr(weight_A.t())    # QA: [in_dim, r],  RA: [r, r]

        # 2. SVD on tiny (r x r) matrix
        M = RB @ RA.t()                          # [r, r]
        U, S, Vh = torch.linalg.svd(M)           # U: [r, r], S: [r], Vh: [r, r]

        # 3. Energy retained
        energy_full = torch.sum(S ** 2).item()
        energy_k = torch.sum(S[:target_rank] ** 2).item()
        energy_ratio = (energy_k / energy_full) if energy_full > 0 else 1.0
        total_energy_ratio.append(energy_ratio)

        # 4. Truncate to target_rank
        Uk = U[:, :target_rank]                  # [r, k]
        Sk = S[:target_rank]                     # [k]
        Vk = Vh[:target_rank, :]                 # [k, r]

        sqrt_Sk = torch.diag(torch.sqrt(Sk))     # [k, k]

        # 5. Reconstruct new factors
        new_B = (QB @ Uk @ sqrt_Sk).to(orig_dtype)          # [out_dim, k]
        new_A = (sqrt_Sk @ Vk @ QA.t()).to(orig_dtype)      # [k, in_dim]

        new_state_dict[key_A] = new_A
        new_state_dict[key_B] = new_B

        # Extract module name for target_modules config
        for mod_name in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
            if mod_name in prefix:
                target_modules.add(mod_name)

    # Calculate average energy retained
    avg_energy = (sum(total_energy_ratio) / len(total_energy_ratio)) * 100 if total_energy_ratio else 0
    print(f"\nCompression complete across {len(new_state_dict)//2} modules.")
    print(f"Average singular energy retained by rank-{target_rank}: {avg_energy:.2f}%")

    # Save new compressed weights
    out_weights = output_dir / "adapter_model.safetensors"
    print(f"Saving compressed weights to {out_weights}...")
    save_file(new_state_dict, str(out_weights))

    # Update and save config
    new_alpha = int(target_rank * scaling_ratio)
    config["r"] = target_rank
    config["lora_alpha"] = new_alpha
    config["target_modules"] = sorted(list(target_modules))

    out_config = output_dir / "adapter_config.json"
    out_config.write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Copy auxiliary files if present (README, etc.)
    for aux in ["README.md"]:
        if (input_dir / aux).is_file():
            shutil.copy2(input_dir / aux, output_dir / aux)

    orig_size_mb = weights_file.stat().st_size / (1024 * 1024)
    new_size_mb = out_weights.stat().st_size / (1024 * 1024)
    reduction = orig_size_mb / new_size_mb if new_size_mb > 0 else 1

    print(f"\n--- Compression Summary ---")
    print(f"  Source: {input_dir.name} (r={source_rank}, alpha={source_alpha}, size={orig_size_mb:.2f} MB)")
    print(f"  Output: {output_dir.name} (r={target_rank}, alpha={new_alpha}, size={new_size_mb:.2f} MB)")
    print(f"  Reduction Ratio: {reduction:.2f}x ({orig_size_mb - new_size_mb:.1f} MB saved)")
    print(f"  Target Modules: {sorted(list(target_modules))}")


def main():
    parser = argparse.ArgumentParser(description="Compress LoRA adapter via SVD.")
    parser.add_argument(
        "--input_dir",
        default=str(ROOT / "models" / "lora_v21_scottweeden_finetuned"),
        help="Path to source LoRA adapter directory",
    )
    parser.add_argument(
        "--output_dir",
        default=str(ROOT / "models" / "lora_v24_rank4_svd"),
        help="Path to save compressed LoRA adapter",
    )
    parser.add_argument("--target_rank", type=int, default=4, help="Target rank (default: 4)")
    parser.add_argument("--attention_only", action="store_true", help="Compress and keep only attention modules")
    args = parser.parse_args()

    compress_lora(
        Path(args.input_dir),
        Path(args.output_dir),
        target_rank=args.target_rank,
        attention_only=args.attention_only,
    )


if __name__ == "__main__":
    main()

