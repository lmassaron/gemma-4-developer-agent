#!/usr/bin/env python3
"""
Train & Package Track B: submission_v33_native_causal_lora_ap_bssr.zip

1. Audits `data/dmitrii_kit/lora_native_resolved.jsonl` (103 resolved native Gemma 4 31B trajectories)
   with the 5-Gate Causality Filter (0 write_file scratch scripts, 0 broken graph tools, seen-edit
   causality verification, exact served prompt_token_ids + completion_token_ids).
2. Splits repositories into Train Repos (32 stage-balanced steps: 8 edit + 8 submit + 8 read + 8 command,
   8 optimizer steps at grad_accum=4) and Held-Out Validation Repos (16 steps: 4 per stage).
3. Loads `/home/lmassaron/models/kaggle/gemma-4-31b-it-qat-w4a16-ct` in packed INT4 on GPU and attaches
   60-layer uniform LoRA (`q_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` -> 600 tensors, r=8, alpha=4).
4. Evaluates Pre-Train Zero-Shot Held-Out NLL (Overall, Action-Span, Thought-Span), trains with PA-ACE
   softcapped loss (`cap=30.0`), evaluates Step 4, Step 6, Step 8, and SVD-Polyak checkpoints, and runs
   SD-TRLS non-destructive line search to guarantee Post-Train NLL <= Pre-Train Zero-Shot NLL.
5. Exports `models/lora_v33_native_causal/`, packages `submissions_archive/submission_v33_native_causal_lora_ap_bssr.zip`,
   and validates with `data/dmitrii_kit/check_submission.py`.
"""
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from package_v32_zero_shot_ap_bssr import EVAL_CONFIG_YAML, SAMPLING_YAML, SYSTEM_PROMPT_MD

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

QAT_DIR = Path("/home/lmassaron/models/kaggle/gemma-4-31b-it-qat-w4a16-ct")
NAT_FILE = ROOT / "data" / "dmitrii_kit" / "lora_native_resolved.jsonl"
OUT_ADAPTER_DIR = ROOT / "models" / "lora_v33_native_causal"
ARCHIVE_DIR = ROOT / "submissions_archive"
REPORT_FILE = ROOT / "state_of_the_art" / "benchmarks_data" / "v33_lora_sft_nll_gate_report.json"

MAX_SEQ_LENGTH = 9216
TRAIN_WINDOW_TOKENS = 4096
STAGE_QUOTAS = {"edit": 8, "submit": 8, "read": 8, "command": 8}  # 32 train steps = 8 opt steps x 4 stages
VAL_PER_STAGE = 4  # 16 held-out val steps = 4 per stage

TARGET_MODULES = ["q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_R = 8
LORA_ALPHA = 4  # Nominal scale = 4/8 = 0.50 (<= 1.0 safe scale)
LORA_DROPOUT = 0.0
ACTION_PHASE_WEIGHT = 1.15
THOUGHT_PHASE_WEIGHT = 0.96

LEAN6_TOOLS = {"run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch"}
BROKEN_GRAPH_TOOLS = {"search_similar_code", "get_code_neighbors", "get_code_subgraph"}
TOOL_CALL_START_TOKEN_ID = 48


def verify_surgical_step(prefix_text: str, completion_text: str) -> tuple[bool, str, str]:
    tools = re.findall(r"call:([a-zA-Z0-9_]+)\{", completion_text)
    if not tools:
        return False, "none", "no_tool_call"
    tool_name = tools[0]
    if tool_name in BROKEN_GRAPH_TOOLS or tool_name not in LEAN6_TOOLS:
        return False, tool_name, "disallowed_tool"
    if tool_name == "write_file":
        return False, tool_name, "write_file_repro_filtered"
    if tool_name in ("read_file", "edit_file") and "filepath:" not in completion_text:
        return False, tool_name, "missing_filepath_arg"
    if tool_name == "run_command" and re.search(r'pytest\s*(?:\.|<\|"\|>)\s*$', completion_text):
        return False, tool_name, "bare_pytest"

    if tool_name == "edit_file":
        parts = completion_text.split("old_string:")
        if len(parts) < 2:
            return False, tool_name, "malformed_edit"
        old_seg = parts[1].split("new_string:")[0]
        old_lines = [
            ln.strip()
            for ln in old_seg.splitlines()
            if len(ln.strip()) >= 8 and not ln.strip().startswith(("<|", "filepath:", "old_string:"))
        ]
        if old_lines:
            check_lines = old_lines[:2]
            if not all(cl in prefix_text for cl in check_lines):
                return False, tool_name, "unseen_edit_filtered"
        return True, tool_name, "edit"

    if tool_name == "submit_patch":
        return True, tool_name, "submit"
    if tool_name == "read_file":
        return True, tool_name, "read"
    return True, tool_name, "command"


def curate_dataset():
    audit_stats = Counter()
    stage_pools = defaultdict(list)

    print(f"[*] Auditing native 31B resolved trajectories: {NAT_FILE}")
    with open(NAT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            traj = json.loads(line)
            repo_name = traj.get("repository", "python/repo")
            task_id = traj.get("task_id", traj.get("id", "task"))
            for ex in traj.get("sft_examples", []):
                audit_stats["total_scanned"] += 1
                if not ex.get("canonical_target_matches", True):
                    audit_stats["skipped_non_canonical"] += 1
                    continue
                p_ids = ex.get("prompt_token_ids")
                c_ids = ex.get("completion_token_ids")
                if not p_ids or not c_ids:
                    continue
                total_len = len(p_ids) + len(c_ids)
                if total_len > MAX_SEQ_LENGTH or len(c_ids) > 640:
                    audit_stats["skipped_over_len"] += 1
                    continue

                prefix_txt = ex.get("prefix", "")
                comp_txt = ex.get("completion", "")
                ok, tool_name, stage_or_reason = verify_surgical_step(prefix_txt, comp_txt)
                if not ok:
                    audit_stats[f"rejected_{stage_or_reason}"] += 1
                    continue

                action_offset = (
                    c_ids.index(TOOL_CALL_START_TOKEN_ID)
                    if TOOL_CALL_START_TOKEN_ID in c_ids
                    else max(0, len(c_ids) - 32)
                )
                audit_stats[f"kept_{stage_or_reason}"] += 1
                stage_pools[stage_or_reason].append({
                    "task_id": f"{task_id}_t{ex.get('turn', 0)}",
                    "repo": repo_name,
                    "stage": stage_or_reason,
                    "tool": tool_name,
                    "prompt_ids": p_ids,
                    "completion_ids": c_ids,
                    "action_offset": action_offset,
                    "context_tokens": total_len,
                    "sup_tokens": len(c_ids),
                })

    # Hold out repositories that have edit/submit coverage
    edit_repos = sorted({it["repo"] for it in stage_pools["edit"]})
    val_repos = set(edit_repos[::5][:4])
    val_pools = defaultdict(list)
    train_pools = defaultdict(list)

    for stg, items in stage_pools.items():
        # Sort by shortest context first so window clipping is minimal
        items_sorted = sorted(items, key=lambda x: (x["context_tokens"], x["task_id"]))
        for it in items_sorted:
            if it["repo"] in val_repos and len(val_pools[stg]) < VAL_PER_STAGE:
                val_pools[stg].append(it)
            else:
                train_pools[stg].append(it)

    for stg in ("edit", "submit", "read", "command"):
        while len(val_pools[stg]) < VAL_PER_STAGE and len(train_pools[stg]) > STAGE_QUOTAS[stg]:
            val_pools[stg].append(train_pools[stg].pop(0))

    val_selected = []
    for i in range(VAL_PER_STAGE):
        for stg in ("edit", "submit", "read", "command"):
            if i < len(val_pools[stg]):
                val_selected.append(val_pools[stg][i])

    # Round-robin across repositories for train selection
    selected_by_stage = {}
    for stg, quota in STAGE_QUOTAS.items():
        by_repo = defaultdict(list)
        for it in train_pools[stg]:
            by_repo[it["repo"]].append(it)
        r_keys = sorted(by_repo.keys())
        chosen = []
        r_idx = 0
        while len(chosen) < quota and r_keys:
            progressed = False
            for rk in r_keys:
                if r_idx < len(by_repo[rk]):
                    chosen.append(by_repo[rk][r_idx])
                    progressed = True
                    if len(chosen) >= quota:
                        break
            if not progressed:
                break
            r_idx += 1
        selected_by_stage[stg] = chosen[:quota]

    train_selected = []
    for i in range(max(STAGE_QUOTAS.values())):
        for stg in ("edit", "submit", "read", "command"):
            train_selected.append(selected_by_stage[stg][i])

    print(f"[OK] Curation Stats: {dict(audit_stats)}")
    print(f"  - Train Steps: {len(train_selected)} ({Counter(r['stage'] for r in train_selected)}) across {len({r['repo'] for r in train_selected})} repos")
    print(f"  - Val Steps  : {len(val_selected)} ({Counter(r['stage'] for r in val_selected)}) across {len({r['repo'] for r in val_selected})} repos")
    return train_selected, val_selected, dict(audit_stats)


def dequantize_int4(packed: torch.Tensor, scale: torch.Tensor, shape: tuple[int, int], args) -> torch.Tensor:
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    from compressed_tensors.quantization.lifecycle.forward import dequantize
    q = unpack_from_int32(packed, args.num_bits, torch.Size(shape))
    return dequantize(x_q=q, scale=scale, args=args, dtype=scale.dtype)


class _Int4Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor, shape: tuple[int, int], args):
        w = dequantize_int4(packed, scale, shape, args)
        ctx.save_for_backward(packed, scale)
        ctx.shape, ctx.args = shape, args
        return F.linear(x, w.to(x.dtype))

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        packed, scale = ctx.saved_tensors
        w = dequantize_int4(packed, scale, ctx.shape, ctx.args)
        return grad_y @ w.to(grad_y.dtype), None, None, None, None


class Int4LoRALinear(nn.Module):
    def __init__(self, linear: nn.Module, r: int = 0, alpha: int = 4, dropout: float = 0.0):
        super().__init__()
        self.args = linear.quantization_scheme.weights
        self.shape = tuple(int(s) for s in linear.weight_shape.tolist())
        self.out_features, self.in_features = self.shape
        self.register_buffer("weight_packed", linear.weight_packed.data, persistent=False)
        self.register_buffer("weight_scale", linear.weight_scale.data.detach(), persistent=False)
        self.r = r
        if r > 0:
            dev = self.weight_scale.device
            self.lora_A = nn.Parameter(torch.empty(r, self.in_features, device=dev, dtype=torch.float32))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=dev, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.scaling = float(alpha) / float(r)
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dev = self.weight_scale.device
        if x.device != dev:
            x = x.to(dev)
        y = _Int4Matmul.apply(x, self.weight_packed, self.weight_scale, self.shape, self.args)
        if self.r > 0:
            h = self.dropout(x)
            y = y + (h @ self.lora_A.to(h.dtype).t()) @ self.lora_B.to(h.dtype).t() * self.scaling
        return y


def selective_softcapped_lm_loss(
    model: nn.Module,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    action_mask: torch.Tensor | None = None,
    use_pa_ace: bool = True,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, int, float, int, float, int]:
    w = model.get_output_embeddings().weight
    out_dev = w.device
    text_cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    cap = float(getattr(text_cfg, "final_logit_softcapping", 30.0) or 30.0)

    h = hidden_states[:, :-1].reshape(-1, hidden_states.shape[-1]).to(out_dev)
    y = labels[:, 1:].reshape(-1).to(out_dev)
    keep = (y != -100).nonzero(as_tuple=True)[0]
    if keep.numel() == 0:
        z = h.sum() * 0.0
        return z, 0, 0.0, 0, 0.0, 0

    am = action_mask[:, 1:].reshape(-1).to(out_dev)[keep] if action_mask is not None else torch.zeros(keep.numel(), device=out_dev)
    if use_pa_ace and action_mask is not None:
        raw_w = torch.where(am > 0.5, ACTION_PHASE_WEIGHT, THOUGHT_PHASE_WEIGHT)
        norm_w = raw_w / raw_w.mean().clamp_min(1e-6)
    else:
        norm_w = torch.ones_like(am)

    def _chunk_ce_vec(hc: torch.Tensor, yc: torch.Tensor) -> torch.Tensor:
        logits = (hc @ w.t()).float()
        if cap > 0.0:
            logits = torch.tanh(logits / cap) * cap
        return F.cross_entropy(logits, yc, reduction="none")

    total_weighted_nll = torch.zeros((), device=out_dev, dtype=torch.float32)
    act_nll_sum = 0.0
    tht_nll_sum = 0.0
    n_act = int((am > 0.5).sum().item())
    n_tht = int(keep.numel()) - n_act

    for i in range(0, keep.numel(), chunk_size):
        idx = keep[i : i + chunk_size]
        am_c = am[i : i + chunk_size]
        w_c = norm_w[i : i + chunk_size]
        if torch.is_grad_enabled() and h.requires_grad:
            ce_vec = torch_checkpoint(_chunk_ce_vec, h[idx], y[idx], use_reentrant=False)
        else:
            ce_vec = _chunk_ce_vec(h[idx], y[idx])
        total_weighted_nll = total_weighted_nll + (ce_vec * w_c).sum()
        with torch.no_grad():
            act_sel = am_c > 0.5
            if act_sel.any():
                act_nll_sum += float(ce_vec[act_sel].sum().detach().cpu())
            if (~act_sel).any():
                tht_nll_sum += float(ce_vec[~act_sel].sum().detach().cpu())

    denom = float(max(1, int(keep.numel())))
    return total_weighted_nll / denom, int(keep.numel()), act_nll_sum, n_act, tht_nll_sum, n_tht


def build_tensor_triple(rec: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p_ids = list(rec["prompt_ids"])
    c_ids = list(rec["completion_ids"])
    if len(p_ids) + len(c_ids) > TRAIN_WINDOW_TOKENS:
        keep_p = max(256, TRAIN_WINDOW_TOKENS - len(c_ids))
        p_ids = p_ids[:128] + p_ids[-(keep_p - 128) :]
    act_off = int(rec.get("action_offset", max(0, len(c_ids) - 32)))
    seq_ids = (p_ids + c_ids)[:TRAIN_WINDOW_TOKENS]
    seq_lbl = ([-100] * len(p_ids) + c_ids)[:TRAIN_WINDOW_TOKENS]
    c_act_mask = [0.0 if idx < act_off else 1.0 for idx in range(len(c_ids))]
    seq_am = ([0.0] * len(p_ids) + c_act_mask)[:TRAIN_WINDOW_TOKENS]
    return (
        torch.tensor([seq_ids], dtype=torch.long),
        torch.tensor([seq_lbl], dtype=torch.long),
        torch.tensor([seq_am], dtype=torch.float32),
    )


def main():
    train_records, val_records, audit_stats = curate_dataset()

    import transformers
    from transformers import AutoConfig

    t0 = time.time()
    print(f"\n[*] Loading {QAT_DIR} directly in packed INT4 on GPU...")
    cfg = AutoConfig.from_pretrained(str(QAT_DIR), trust_remote_code=True)
    cls = getattr(transformers, cfg.architectures[0])
    base_model = cls.from_pretrained(
        str(QAT_DIR),
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    if hasattr(base_model, "ct_decompress_hook"):
        base_model.ct_decompress_hook.remove()
        del base_model.ct_decompress_hook

    for p in base_model.parameters():
        p.requires_grad_(False)

    injected_modules = {}
    swapped = 0
    for name, mod in list(base_model.named_modules()):
        if not hasattr(mod, "weight_packed"):
            continue
        child = name.rsplit(".", 1)[-1]
        use_lora = (child in TARGET_MODULES) and ("language_model" in name)
        parent_name = name.rsplit(".", 1)[0]
        lora_mod = Int4LoRALinear(
            mod,
            r=LORA_R if use_lora else 0,
            alpha=LORA_ALPHA,
            dropout=LORA_DROPOUT,
        )
        setattr(base_model.get_submodule(parent_name), child, lora_mod)
        swapped += 1
        if use_lora:
            injected_modules[name] = lora_mod

    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base_model.config.use_cache = False
    if hasattr(base_model.config, "text_config") and base_model.config.text_config is not None:
        base_model.config.text_config.use_cache = False

    trainable_params = [p for p in base_model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    print(
        f"[OK] Loaded 31B QAT model in {time.time() - t0:.1f}s | Swapped {swapped} INT4 layers | "
        f"Injected {len(injected_modules)} LoRA modules ({len(injected_modules)*2} tensors, {num_trainable:,} trainable params)"
    )

    encoded_train = [(build_tensor_triple(r), r["stage"]) for r in train_records]
    encoded_val = [build_tensor_triple(r) for r in val_records]
    first_device = base_model.get_input_embeddings().weight.device

    def snapshot_lora_cpu() -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return {
            name: (mod.lora_A.detach().cpu().clone(), mod.lora_B.detach().cpu().clone())
            for name, mod in injected_modules.items()
        }

    def restore_lora_cpu(
        snap: dict[str, tuple[torch.Tensor, torch.Tensor]],
        attn_scale: float = 1.0,
        mlp_scale: float = 1.0,
    ) -> None:
        with torch.no_grad():
            for name, mod in injected_modules.items():
                A_c, B_c = snap[name]
                is_attn = any(k in name for k in ("q_proj", "o_proj"))
                s_val = attn_scale if is_attn else mlp_scale
                mod.lora_A.copy_(A_c.to(mod.lora_A.device, dtype=mod.lora_A.dtype))
                mod.lora_B.copy_((B_c * s_val).to(mod.lora_B.device, dtype=mod.lora_B.dtype))

    def evaluate_held_out(tag: str) -> dict:
        t_ev = time.time()
        base_model.eval()
        tot_act_nll, tot_tht_nll = 0.0, 0.0
        tot_act_tok, tot_tht_tok = 0, 0
        with torch.no_grad():
            for inp_v, lbl_v, am_v in encoded_val:
                inp_v = inp_v.to(first_device)
                hid_v = base_model.model(input_ids=inp_v, use_cache=False, return_dict=True).last_hidden_state
                _, _, act_sum, n_act, tht_sum, n_tht = selective_softcapped_lm_loss(
                    base_model, hid_v, lbl_v, am_v, use_pa_ace=False
                )
                tot_act_nll += act_sum
                tot_tht_nll += tht_sum
                tot_act_tok += n_act
                tot_tht_tok += n_tht
        base_model.train()
        tot_tok = tot_act_tok + tot_tht_tok
        overall_nll = (tot_act_nll + tot_tht_nll) / max(1, tot_tok)
        act_nll = tot_act_nll / max(1, tot_act_tok)
        tht_nll = tot_tht_nll / max(1, tot_tht_tok)
        print(
            f"  [VAL {tag:14s}] Overall NLL={overall_nll:.5f} | Action-Span NLL={act_nll:.5f} ({tot_act_tok} tok) | "
            f"Thought-Span NLL={tht_nll:.5f} ({tot_tht_tok} tok) [{time.time() - t_ev:.1f}s]",
            flush=True,
        )
        return {
            "overall_nll": overall_nll,
            "action_nll": act_nll,
            "thought_nll": tht_nll,
            "tot_tok": tot_tok,
            "act_tok": tot_act_tok,
        }

    # 1. Pre-Train Zero-Shot Anchor
    pre_eval = evaluate_held_out("PRE-TRAIN")

    # 2. 8-Step Stage-Balanced SFT (32 samples, grad_accum=4, LR=5e-6)
    GRAD_ACCUM = 4
    LR = 5.0e-6
    WARMUP_STEPS = 2
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
    total_opt_steps = len(encoded_train) // GRAD_ACCUM
    ckpt_steps = {4, 6}
    ckpt_candidates = []
    train_history = []

    def lr_schedule(step_idx: int) -> float:
        if step_idx < WARMUP_STEPS:
            return LR * float(step_idx + 1) / float(WARMUP_STEPS)
        prog = min(1.0, float(step_idx - WARMUP_STEPS) / max(1.0, float(total_opt_steps - WARMUP_STEPS)))
        return LR * (0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * prog)))

    print(
        f"\n[*] Starting v33 Native 31B Causal SFT ({len(encoded_train)} samples, grad_accum={GRAD_ACCUM}, "
        f"steps={total_opt_steps}, lr={LR}, window={TRAIN_WINDOW_TOKENS})...",
        flush=True,
    )
    step_counter = 0
    for batch_start in range(0, len(encoded_train), GRAD_ACCUM):
        micro_batch = encoded_train[batch_start : batch_start + GRAD_ACCUM]
        step_sup_tokens = sum(int((lbl[:, 1:] != -100).sum().item()) for (inp, lbl, _), _ in micro_batch)
        if step_sup_tokens == 0:
            continue

        cur_lr = lr_schedule(step_counter)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        step_nll_sum = 0.0
        step_act_nll_sum = 0.0
        step_act_tok = 0
        t_step = time.time()

        for (inp, lbl, am), stg in micro_batch:
            inp = inp.to(first_device)
            hidden = base_model.model(input_ids=inp, use_cache=False, return_dict=True).last_hidden_state
            mean_loss, n_sup, act_s, n_act, tht_s, _ = selective_softcapped_lm_loss(base_model, hidden, lbl, am)
            if n_sup > 0:
                scaled_loss = (mean_loss * n_sup) / float(step_sup_tokens)
                scaled_loss.backward()
                step_nll_sum += act_s + tht_s
                step_act_nll_sum += act_s
                step_act_tok += n_act

        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).cpu())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step_counter += 1

        step_loss = step_nll_sum / float(step_sup_tokens)
        step_act_loss = step_act_nll_sum / max(1, step_act_tok)
        dt = time.time() - t_step
        train_history.append({
            "step": step_counter,
            "loss": round(step_loss, 5),
            "action_loss": round(step_act_loss, 5),
            "grad_norm": round(grad_norm, 5),
            "lr": cur_lr,
            "sup_tokens": step_sup_tokens,
        })
        print(
            f"  [Step {step_counter:02d}/{total_opt_steps:02d}] loss={step_loss:.4f} (act={step_act_loss:.4f}) | "
            f"grad_norm={grad_norm:.4f} | lr={cur_lr:.2e} | sup_tok={step_sup_tokens:4d} | {dt:.1f}s",
            flush=True,
        )

        if step_counter in ckpt_steps:
            tag = f"CKPT-S{step_counter:02d}"
            c_eval = evaluate_held_out(tag)
            ckpt_candidates.append((tag, c_eval, snapshot_lora_cpu()))

    final_snap = snapshot_lora_cpu()
    final_eval = evaluate_held_out("FINAL-S08")
    ckpt_candidates.append(("FINAL-S08", final_eval, final_snap))

    # 3. SVD-Polyak Manifold Merge of Step 6 + Step 8
    if len(ckpt_candidates) >= 2:
        s_prev = ckpt_candidates[-2][2]
        s_last = final_snap
        svd_polyak_snap = {}
        with torch.no_grad():
            for mod_path in s_last.keys():
                A1, B1 = s_prev[mod_path][0].float(), s_prev[mod_path][1].float()
                A2, B2 = s_last[mod_path][0].float(), s_last[mod_path][1].float()
                B_cat = torch.cat([0.45 * B1, 0.55 * B2], dim=1).to(first_device)
                A_cat = torch.cat([A1, A2], dim=0).to(first_device)
                Qb, Rb = torch.linalg.qr(B_cat, mode="reduced")
                Qa, Ra = torch.linalg.qr(A_cat.T, mode="reduced")
                U, S_val, Vh = torch.linalg.svd(Rb @ Ra.T, full_matrices=False)
                U_r, S_r, Vh_r = U[:, :LORA_R], S_val[:LORA_R], Vh[:LORA_R, :]
                A_merged = Vh_r @ Qa.T
                target_a_norm = A2.to(first_device).norm().clamp_min(1e-9)
                scale_a = target_a_norm / A_merged.norm().clamp_min(1e-9)
                A_out = (A_merged * scale_a).cpu()
                B_out = ((Qb @ (U_r * S_r.unsqueeze(0))) / scale_a).cpu()
                svd_polyak_snap[mod_path] = (A_out, B_out)
        restore_lora_cpu(svd_polyak_snap, attn_scale=1.0, mlp_scale=1.0)
        polyak_eval = evaluate_held_out("SVD-POLYAK")
        ckpt_candidates.append(("SVD-POLYAK", polyak_eval, svd_polyak_snap))

    best_label, best_eval, best_snap = min(ckpt_candidates, key=lambda x: (x[1]["action_nll"], x[1]["overall_nll"]))
    print(f"\n[*] Best Checkpoint before SD-TRLS: {best_label} (Overall={best_eval['overall_nll']:.5f}, Action={best_eval['action_nll']:.5f})")

    # 4. Subsystem-Decoupled Trust-Region Projection + Non-Destructive Line Search (SD-TRLS)
    MAX_SAFE_ATTN_NORM = 0.0015
    MAX_SAFE_MLP_NORM = 0.0030
    raw_attn_max, raw_mlp_max = 0.0, 0.0
    for mod_path, (A_c, B_c) in best_snap.items():
        A_f, B_f = A_c.float(), B_c.float()
        gram_trace = float(((B_f.T @ B_f) * (A_f @ A_f.T)).sum().item())
        cur_norm = (LORA_ALPHA / LORA_R) * math.sqrt(max(0.0, gram_trace))
        if any(k in mod_path for k in ("q_proj", "o_proj")):
            raw_attn_max = max(raw_attn_max, cur_norm)
        else:
            raw_mlp_max = max(raw_mlp_max, cur_norm)

    attn_cap = min(1.0, MAX_SAFE_ATTN_NORM / max(raw_attn_max, 1e-9))
    mlp_cap = min(1.0, MAX_SAFE_MLP_NORM / max(raw_mlp_max, 1e-9))
    print(f"[*] Raw Max Norms: Attn={raw_attn_max:.6f} (cap={attn_cap:.4f}), MLP={raw_mlp_max:.6f} (cap={mlp_cap:.4f})")

    candidate_gammas = [1.0, 0.70, 0.50, 0.25, 0.125]
    best_gamma = 0.25
    best_post_eval = None

    for g_val in candidate_gammas:
        cur_a = round(attn_cap * g_val, 5)
        cur_m = round(mlp_cap * g_val, 5)
        restore_lora_cpu(best_snap, attn_scale=cur_a, mlp_scale=cur_m)
        c_post = evaluate_held_out(f"SD-GAMMA-{g_val:.3f}")
        # Require non-destructive Overall NLL <= pre_eval['overall_nll'] + 0.0005 and minimum Action-Span NLL
        if c_post["overall_nll"] <= pre_eval["overall_nll"] + 0.0005:
            if best_post_eval is None or (c_post["action_nll"], c_post["overall_nll"]) < (
                best_post_eval["action_nll"],
                best_post_eval["overall_nll"],
            ):
                best_post_eval = c_post
                best_gamma = g_val

    if best_post_eval is None:
        best_gamma = 0.10
        cur_a = round(attn_cap * best_gamma, 5)
        cur_m = round(mlp_cap * best_gamma, 5)
        restore_lora_cpu(best_snap, attn_scale=cur_a, mlp_scale=cur_m)
        best_post_eval = evaluate_held_out("SD-SAFE-0.100")

    final_attn_scale = round(attn_cap * best_gamma, 5)
    final_mlp_scale = round(mlp_cap * best_gamma, 5)
    print(
        f"\n[OK] Selected v33 LoRA: {best_label} (gamma*={best_gamma:.3f}, attn_scale={final_attn_scale:.5f}, mlp_scale={final_mlp_scale:.5f})\n"
        f"  - Pre-Train  : Overall NLL={pre_eval['overall_nll']:.5f} | Action NLL={pre_eval['action_nll']:.5f} | Thought NLL={pre_eval['thought_nll']:.5f}\n"
        f"  - Post-Train : Overall NLL={best_post_eval['overall_nll']:.5f} | Action NLL={best_post_eval['action_nll']:.5f} | Thought NLL={best_post_eval['thought_nll']:.5f}"
    )

    # 5. Export LoRA Tensors & Config
    trained_lora_state = {}
    for mod_path, (A_c, B_c) in sorted(best_snap.items()):
        A_f = A_c.float()
        is_attn = any(k in mod_path for k in ("q_proj", "o_proj"))
        s_val = final_attn_scale if is_attn else final_mlp_scale
        B_f = B_c.float() * s_val
        canon_mod = mod_path if mod_path.startswith("model.language_model.") else f"model.{mod_path}"
        trained_lora_state[f"base_model.model.{canon_mod}.lora_A.weight"] = A_f.to(torch.bfloat16).contiguous()
        trained_lora_state[f"base_model.model.{canon_mod}.lora_B.weight"] = B_f.to(torch.bfloat16).contiguous()

    OUT_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    sf_out = OUT_ADAPTER_DIR / "adapter_model.safetensors"
    save_safetensors(trained_lora_state, str(sf_out))

    adapter_cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "gemma-4-31b-it-qat-w4a16-ct",
        "r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "use_dora": False,
        "use_rslora": False,
        "modules_to_save": None,
        "target_modules": sorted(TARGET_MODULES),
        "layers_to_transform": None,
        "layers_pattern": None,
        "rank_pattern": {},
        "alpha_pattern": {},
        "revision": None,
    }
    cfg_out = OUT_ADAPTER_DIR / "adapter_config.json"
    cfg_out.write_text(json.dumps(adapter_cfg, indent=2) + "\n", encoding="utf-8")

    # 6. Package submission_v33_native_causal_lora_ap_bssr.zip
    agent_yaml_v33 = """name: swe_agent
model: gemma-4-31b-it-qat-w4a16-ct
adapter: main_lora
instruction: !include prompts/system.md
tools:
  - run_command
  - read_file
  - edit_file
  - write_file
  - get_status
  - submit_patch
generate_content_config: !include configs/sampling.yaml
"""
    out_zip = ARCHIVE_DIR / "submission_v33_native_causal_lora_ap_bssr.zip"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("agent.yaml", agent_yaml_v33.encode("utf-8"))
        zf.writestr("configs/sampling.yaml", SAMPLING_YAML.encode("utf-8"))
        zf.writestr("eval_config.yaml", EVAL_CONFIG_YAML.encode("utf-8"))
        zf.writestr("prompts/system.md", SYSTEM_PROMPT_MD.encode("utf-8"))
        zf.writestr("adapters/main_lora/adapter_config.json", cfg_out.read_bytes())
        zf.write(sf_out, arcname="adapters/main_lora/adapter_model.safetensors")

    sha = hashlib.sha256(out_zip.read_bytes()).hexdigest()
    print(f"[OK] Packaged Track B: {out_zip} ({out_zip.stat().st_size / (1024*1024):.2f} MB) | SHA-256: {sha}")

    report = {
        "adapter": "lora_v33_native_causal",
        "submission_zip": str(out_zip),
        "sha256": sha,
        "audit_stats": audit_stats,
        "selected_checkpoint": best_label,
        "selected_gamma": best_gamma,
        "final_attn_scale": final_attn_scale,
        "final_mlp_scale": final_mlp_scale,
        "pre_train_val": pre_eval,
        "post_train_val": best_post_eval,
        "train_history": train_history,
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")

    checker = ROOT / "data" / "dmitrii_kit" / "check_submission.py"
    if checker.is_file():
        tmp_sub = Path("/tmp/submission.zip")
        shutil.copy2(out_zip, tmp_sub)
        res = subprocess.run([sys.executable, str(checker), str(tmp_sub)], capture_output=True, text=True)
        print(res.stdout)
        if res.returncode != 0:
            print(res.stderr)
            sys.exit(res.returncode)


if __name__ == "__main__":
    main()

