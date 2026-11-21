#!/usr/bin/env python3
"""
Train & Package v46: Patchsmith-Aligned 6-Tool Native Causal LoRA on data/dmitrii_kit.

Key improvements over v33:
1. Full 6-tool Patchsmith (exp021/v45) prompt & schema alignment via exact system-header
   token swap while preserving served completion_token_ids verbatim.
2. Strict surgical data curation from lora_native_resolved.jsonl (next-turn tool error
   filtering, scratch write_file rejection, duplicate command rejection, no truncation
   up to MAX_SEQ_LENGTH=10240).
3. Repo-level train/val split (0 repo overlap with requests/fastapi/rich/sympy) and
   meaningful budget (384 curated steps -> 48 optimizer steps at grad_accum=8).
4. Clean r=16, lora_alpha=8 (scale 0.50 <= 1.0) across all 410 language_model linears
   (excluding vision_tower via explicit target_modules list) with checkpoint + scale sweep.
"""
import argparse
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
from safetensors.torch import save_file as save_safetensors
from torch.utils.checkpoint import checkpoint as torch_checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from package_v45_candidate import V45_EVAL_YAML, V45_SAMPLING_YAML, V45_SYSTEM_MD
from run_local_benchmark_leaderboard import ADK_TOOL_SCHEMAS

SEED = 42
NAT_FILE = ROOT / "data" / "dmitrii_kit" / "lora_native_resolved.jsonl"
QAT_DIR = Path("/home/lmassaron/models/kaggle/gemma-4-31b-it-qat-w4a16-ct")
OUT_ADAPTER_DIR = ROOT / "models" / "lora_v46_patchsmith"
OUT_ZIP = ROOT / "submission_v46_patchsmith_lora.zip"
REPORT_FILE = ROOT / "state_of_the_art" / "benchmarks_data" / "v46_lora_training_report.json"

MAX_SEQ_LENGTH = 10240
MAX_COMPLETION_TOKENS = 1536
LORA_R = 16
LORA_ALPHA = 8  # scale = 0.50 (safe <= 1.0); checkpoints also allow scale sweep
LORA_DROPOUT = 0.0
GRAD_ACCUM = 8
MAX_TRAIN_STEPS = 384  # 48 optimizer steps
VAL_TARGET_STEPS = 64  # balanced held-out validation slice across held-out repos
LR = 1.5e-5
WEIGHT_DECAY = 0.01
WARMUP_OPT_STEPS = 4
ACTION_PHASE_WEIGHT = 1.25
THOUGHT_PHASE_WEIGHT = 0.90

LEAN6_TOOLS = {"run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch"}
TOOL_CALL_START_TOKEN_ID = 48
TARGET_PROJS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
LANG_MOD_PAT = re.compile(
    r"^model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$"
)

V46_AGENT_YAML = """name: patchsmith
model: gemma-4-31b-it-qat-w4a16-ct
adapter: main_lora
description: Repairs implementation with bounded navigation, native edits and reserved verification.
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


def build_patchsmith_sys_prefix_ids() -> list[int]:
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained("google/gemma-4-31B-it")
    tok = proc.tokenizer
    sys_str = proc.apply_chat_template(
        [{"role": "system", "content": V45_SYSTEM_MD}, {"role": "user", "content": "__SPLIT__"}],
        tools=ADK_TOOL_SCHEMAS,
        enable_thinking=True,
        tokenize=False,
    )
    sys_ids = tok.encode(sys_str, add_special_tokens=False)
    cut = sys_ids.index(106) + 2  # keep up through <turn|>\n
    prefix_ids = sys_ids[:cut]
    assert prefix_ids[-2:] == [106, 107], f"Unexpected sys prefix end tokens: {prefix_ids[-4:]}"
    return prefix_ids


def curate_dataset(sys_prefix_ids: list[int]):
    stats = Counter()
    all_kept = []

    with open(NAT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            traj = json.loads(line)
            repo = traj.get("repository", "python/repo")
            task_id = traj.get("task_id", traj.get("id", "task"))
            exs = traj.get("sft_examples", [])
            seen_cmds = set()

            for idx, ex in enumerate(exs):
                stats["total_scanned"] += 1
                if not ex.get("canonical_target_matches", True):
                    stats["rej_non_canonical"] += 1
                    continue
                p_ids = ex.get("prompt_token_ids") or []
                c_ids = ex.get("completion_token_ids") or []
                if not p_ids or not c_ids or len(c_ids) > MAX_COMPLETION_TOKENS:
                    stats["rej_empty_or_long_comp"] += 1
                    continue
                tcs = (ex.get("response_message") or {}).get("tool_calls") or []
                if not tcs:
                    stats["rej_no_tool_call"] += 1
                    continue
                fn = tcs[0]["function"]["name"]
                if fn not in LEAN6_TOOLS:
                    stats["rej_bad_tool"] += 1
                    continue

                args_raw = tcs[0]["function"].get("arguments", "{}")
                try:
                    a = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except Exception:
                    stats["rej_bad_args"] += 1
                    continue

                next_resp = ""
                if idx + 1 < len(exs):
                    for m in reversed(exs[idx + 1]["request"]["messages"]):
                        if m.get("role") == "tool":
                            next_resp = str(m.get("content", ""))
                            break

                if fn == "write_file":
                    fp = str(a.get("filepath") or a.get("path") or "")
                    if fp.startswith("/tmp") or any(k in fp.lower() for k in ("repro", "debug", "test_", "_test.py", "check_")):
                        stats["rej_write_scratch"] += 1
                        continue
                    if '"status": "error"' in next_resp.lower():
                        stats["rej_write_err"] += 1
                        continue

                if fn == "edit_file":
                    if any(k in next_resp.lower() for k in ('"status": "error"', "not found", "no match", "multiple matches", "error:")):
                        stats["rej_edit_err"] += 1
                        continue
                    old_s = str(a.get("old_string") or "")
                    old_lines = [ln.strip() for ln in old_s.splitlines() if len(ln.strip()) >= 8]
                    if old_lines and not all(cl in ex.get("prefix", "") for cl in old_lines[:2]):
                        stats["rej_unseen_edit"] += 1
                        continue

                if fn == "read_file":
                    if any(k in next_resp.lower() for k in ("error", "not found", "no such file", "does not exist")):
                        stats["rej_read_err"] += 1
                        continue

                if fn == "run_command":
                    cmd = str(a.get("command") or "").strip()
                    if cmd in seen_cmds or "curl " in cmd or "pip install" in cmd:
                        stats["rej_cmd_dup_or_forbidden"] += 1
                        continue
                    seen_cmds.add(cmd)
                    if '"exit_code": 1' in next_resp and '"stdout": ""' in next_resp:
                        stats["rej_cmd_empty_fail"] += 1
                        continue
                    if re.search(r"^(?:python3?\s+-m\s+)?pytest\s*(?:\.|tests/?)?\s*$", cmd):
                        stats["rej_bare_pytest"] += 1
                        continue

                if fn == "submit_patch":
                    if '"patch_size": 0' in next_resp or '"files_changed": 0' in next_resp:
                        stats["rej_submit_empty"] += 1
                        continue

                # Swap native system+tools header with 6-tool Patchsmith system prefix
                old_cut = p_ids.index(106) + 2
                swapped_p = sys_prefix_ids + p_ids[old_cut:]
                tot_len = len(swapped_p) + len(c_ids)
                if tot_len > MAX_SEQ_LENGTH:
                    stats["rej_over_max_seq"] += 1
                    continue

                stage = {
                    "edit_file": "edit",
                    "submit_patch": "submit",
                    "read_file": "read",
                    "run_command": "command",
                    "write_file": "edit",
                    "get_status": "command",
                }[fn]
                action_offset = (
                    c_ids.index(TOOL_CALL_START_TOKEN_ID)
                    if TOOL_CALL_START_TOKEN_ID in c_ids
                    else max(0, len(c_ids) - 32)
                )
                stats[f"kept_{stage}"] += 1
                all_kept.append({
                    "task_id": f"{task_id}_t{ex.get('turn', idx)}",
                    "repo": repo,
                    "stage": stage,
                    "tool": fn,
                    "prompt_ids": swapped_p,
                    "completion_ids": c_ids,
                    "action_offset": action_offset,
                    "context_tokens": tot_len,
                    "sup_tokens": len(c_ids),
                })

    # Repo-level deterministic split: pick ~20% of repos (covering edit+submit) for validation
    repo_stages = defaultdict(set)
    for it in all_kept:
        repo_stages[it["repo"]].add(it["stage"])
    rich_repos = sorted([r for r, stgs in repo_stages.items() if {"edit", "submit"}.issubset(stgs)])
    rng = random.Random(SEED)
    rng.shuffle(rich_repos)
    val_repos = set(rich_repos[: max(8, len(rich_repos) // 5)])

    val_by_stage = defaultdict(list)
    train_by_stage = defaultdict(list)
    for it in all_kept:
        if it["repo"] in val_repos:
            val_by_stage[it["stage"]].append(it)
        else:
            train_by_stage[it["stage"]].append(it)

    # Select balanced validation slice across held-out repos
    val_quotas = {"edit": 16, "submit": 16, "read": 16, "command": 16}
    val_selected = []
    for stg, q in val_quotas.items():
        pool = list(val_by_stage[stg])
        rng.shuffle(pool)
        val_selected.extend(pool[:q])

    # Select balanced, repo-diverse training set up to MAX_TRAIN_STEPS (divisible by GRAD_ACCUM)
    # Prioritize edit (28%), submit (24%), read (22%), command (26%)
    train_quotas = {"edit": 108, "submit": 92, "read": 84, "command": 100}  # sum = 384
    train_pools = {}
    for stg, q in train_quotas.items():
        by_repo = defaultdict(list)
        for it in train_by_stage[stg]:
            by_repo[it["repo"]].append(it)
        r_keys = sorted(by_repo.keys())
        rng.shuffle(r_keys)
        for rk in r_keys:
            rng.shuffle(by_repo[rk])
        chosen = []
        r_idx = 0
        while len(chosen) < q and r_keys:
            progressed = False
            for rk in r_keys:
                if r_idx < len(by_repo[rk]):
                    chosen.append(by_repo[rk][r_idx])
                    progressed = True
                    if len(chosen) >= q:
                        break
            if not progressed:
                break
            r_idx += 1
        train_pools[stg] = chosen

    train_selected = []
    for stg in ("edit", "submit", "read", "command"):
        train_selected.extend(train_pools[stg])
    rng.shuffle(train_selected)
    usable = (len(train_selected) // GRAD_ACCUM) * GRAD_ACCUM
    train_selected = train_selected[:usable]

    return train_selected, val_selected, dict(stats), sorted(val_repos)


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
    def __init__(self, linear: nn.Module, r: int = 0, alpha: int = 8, dropout: float = 0.0):
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
    use_phase_weights: bool = True,
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
        return h.sum() * 0.0, 0, 0.0, 0, 0.0, 0

    am = action_mask[:, 1:].reshape(-1).to(out_dev)[keep] if action_mask is not None else torch.zeros(keep.numel(), device=out_dev)
    if use_phase_weights and action_mask is not None:
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
    act_nll_sum, tht_nll_sum = 0.0, 0.0
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

    return total_weighted_nll / float(max(1, int(keep.numel()))), int(keep.numel()), act_nll_sum, n_act, tht_nll_sum, n_tht


def build_tensor_triple(rec: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p_ids = list(rec["prompt_ids"])
    c_ids = list(rec["completion_ids"])
    act_off = int(rec.get("action_offset", max(0, len(c_ids) - 32)))
    seq_ids = (p_ids + c_ids)[:MAX_SEQ_LENGTH]
    seq_lbl = ([-100] * len(p_ids) + c_ids)[:MAX_SEQ_LENGTH]
    c_act_mask = [0.0 if idx < act_off else 1.0 for idx in range(len(c_ids))]
    seq_am = ([0.0] * len(p_ids) + c_act_mask)[:MAX_SEQ_LENGTH]
    return (
        torch.tensor([seq_ids], dtype=torch.long),
        torch.tensor([seq_lbl], dtype=torch.long),
        torch.tensor([seq_am], dtype=torch.float32),
    )


def export_adapter_dir(out_dir: Path, snap: dict, target_modules: list[str], scale_mult: float = 1.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {}
    for mod_path, (A_c, B_c) in sorted(snap.items()):
        canon = mod_path if mod_path.startswith("model.language_model.") else f"model.{mod_path}"
        state[f"base_model.model.{canon}.lora_A.weight"] = A_c.float().to(torch.bfloat16).contiguous()
        state[f"base_model.model.{canon}.lora_B.weight"] = (B_c.float() * scale_mult).to(torch.bfloat16).contiguous()
    save_safetensors(state, str(out_dir / "adapter_model.safetensors"))
    cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "gemma-4-31b-it-qat-w4a16-ct",
        "r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "use_dora": False,
        "use_rslora": False,
        "modules_to_save": None,
        "target_modules": sorted(target_modules),
        "layers_to_transform": None,
        "layers_pattern": None,
        "rank_pattern": {},
        "alpha_pattern": {},
        "revision": None,
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def package_submission(adapter_dir: Path, out_zip: Path) -> str:
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel, content in sorted([
            ("agent.yaml", V46_AGENT_YAML),
            ("configs/sampling.yaml", V45_SAMPLING_YAML),
            ("eval_config.yaml", V45_EVAL_YAML),
            ("prompts/system.md", V45_SYSTEM_MD),
        ]):
            info = zipfile.ZipInfo(rel, (1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            zf.writestr(info, content.encode("utf-8"))
        zf.write(adapter_dir / "adapter_config.json", arcname="adapters/main_lora/adapter_config.json")
        zf.write(adapter_dir / "adapter_model.safetensors", arcname="adapters/main_lora/adapter_model.safetensors")
    return hashlib.sha256(out_zip.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Audit curated train/val split without GPU training")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    sys_prefix_ids = build_patchsmith_sys_prefix_ids()
    train_records, val_records, audit_stats, val_repos = curate_dataset(sys_prefix_ids)
    print(f"[Audit] Stats: {audit_stats}")
    print(f"  - Train steps: {len(train_records)} ({Counter(r['stage'] for r in train_records)}) across {len({r['repo'] for r in train_records})} repos")
    print(f"  - Val steps  : {len(val_records)} ({Counter(r['stage'] for r in val_records)}) across {len(val_repos)} held-out repos")
    print(f"  - Max train len: {max(r['context_tokens'] for r in train_records)} | Max val len: {max(r['context_tokens'] for r in val_records)}")

    if args.dry_run:
        return

    import transformers
    from transformers import AutoConfig

    t0 = time.time()
    print(f"\n[*] Loading {QAT_DIR} in packed INT4 on GPU...")
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
    target_module_names = []
    swapped = 0
    for name, mod in list(base_model.named_modules()):
        if not hasattr(mod, "weight_packed"):
            continue
        child = name.rsplit(".", 1)[-1]
        use_lora = bool(LANG_MOD_PAT.match(name)) and (child in TARGET_PROJS)
        parent_name = name.rsplit(".", 1)[0]
        lora_mod = Int4LoRALinear(mod, r=LORA_R if use_lora else 0, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
        setattr(base_model.get_submodule(parent_name), child, lora_mod)
        swapped += 1
        if use_lora:
            injected_modules[name] = lora_mod
            target_module_names.append(name)

    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base_model.config.use_cache = False
    if hasattr(base_model.config, "text_config") and base_model.config.text_config is not None:
        base_model.config.text_config.use_cache = False

    trainable_params = [p for p in base_model.parameters() if p.requires_grad]
    print(
        f"[OK] Loaded in {time.time() - t0:.1f}s | Swapped {swapped} INT4 layers | "
        f"Injected {len(injected_modules)} LoRA modules ({sum(p.numel() for p in trainable_params):,} params)",
        flush=True,
    )

    encoded_train = [build_tensor_triple(r) for r in train_records]
    encoded_val = [build_tensor_triple(r) for r in val_records]
    first_device = base_model.get_input_embeddings().weight.device

    def snapshot_lora_cpu() -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return {n: (m.lora_A.detach().cpu().clone(), m.lora_B.detach().cpu().clone()) for n, m in injected_modules.items()}

    def restore_lora_cpu(snap: dict[str, tuple[torch.Tensor, torch.Tensor]], scale_mult: float = 1.0):
        with torch.no_grad():
            for n, m in injected_modules.items():
                A_c, B_c = snap[n]
                m.lora_A.copy_(A_c.to(m.lora_A.device, dtype=m.lora_A.dtype))
                m.lora_B.copy_((B_c * scale_mult).to(m.lora_B.device, dtype=m.lora_B.dtype))

    def evaluate_held_out(tag: str) -> dict:
        t_ev = time.time()
        base_model.eval()
        tot_act_nll, tot_tht_nll, tot_act_tok, tot_tht_tok = 0.0, 0.0, 0, 0
        with torch.no_grad():
            for inp_v, lbl_v, am_v in encoded_val:
                inp_v = inp_v.to(first_device)
                hid_v = base_model.model(input_ids=inp_v, use_cache=False, return_dict=True).last_hidden_state
                _, _, act_s, n_act, tht_s, n_tht = selective_softcapped_lm_loss(
                    base_model, hid_v, lbl_v, am_v, use_phase_weights=False
                )
                tot_act_nll += act_s
                tot_tht_nll += tht_s
                tot_act_tok += n_act
                tot_tht_tok += n_tht
        base_model.train()
        tot_tok = tot_act_tok + tot_tht_tok
        ov = (tot_act_nll + tot_tht_nll) / max(1, tot_tok)
        ac = tot_act_nll / max(1, tot_act_tok)
        th = tot_tht_nll / max(1, tot_tht_tok)
        print(
            f"  [VAL {tag:14s}] Overall={ov:.5f} | Action={ac:.5f} ({tot_act_tok}t) | Thought={th:.5f} ({tot_tht_tok}t) [{time.time() - t_ev:.1f}s]",
            flush=True,
        )
        return {"overall_nll": ov, "action_nll": ac, "thought_nll": th, "tot_tok": tot_tok, "act_tok": tot_act_tok}

    pre_eval = evaluate_held_out("PRE-TRAIN")

    optimizer = torch.optim.AdamW(trainable_params, lr=LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
    total_opt_steps = len(encoded_train) // GRAD_ACCUM
    ckpt_steps = {16, 32, total_opt_steps}
    ckpt_candidates = []
    train_history = []

    def lr_schedule(step_idx: int) -> float:
        if step_idx < WARMUP_OPT_STEPS:
            return LR * float(step_idx + 1) / float(WARMUP_OPT_STEPS)
        prog = min(1.0, float(step_idx - WARMUP_OPT_STEPS) / max(1.0, float(total_opt_steps - WARMUP_OPT_STEPS)))
        return LR * (0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * prog)))

    print(f"\n[*] Training {total_opt_steps} optimizer steps ({len(encoded_train)} samples, accum={GRAD_ACCUM}, lr={LR})...", flush=True)
    step_counter = 0
    for start in range(0, len(encoded_train), GRAD_ACCUM):
        batch = encoded_train[start : start + GRAD_ACCUM]
        sup_toks = sum(int((lbl[:, 1:] != -100).sum().item()) for _, lbl, _ in batch)
        if sup_toks == 0:
            continue
        cur_lr = lr_schedule(step_counter)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        t_step = time.time()
        nll_sum, act_sum, act_tok = 0.0, 0.0, 0
        for inp, lbl, am in batch:
            inp = inp.to(first_device)
            hid = base_model.model(input_ids=inp, use_cache=False, return_dict=True).last_hidden_state
            loss_val, n_sup, a_s, n_a, t_s, _ = selective_softcapped_lm_loss(base_model, hid, lbl, am)
            if n_sup > 0:
                ((loss_val * n_sup) / float(sup_toks)).backward()
                nll_sum += a_s + t_s
                act_sum += a_s
                act_tok += n_a

        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).cpu())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step_counter += 1

        st_loss = nll_sum / float(sup_toks)
        st_act = act_sum / max(1, act_tok)
        dt = time.time() - t_step
        train_history.append({
            "step": step_counter,
            "loss": round(st_loss, 5),
            "action_loss": round(st_act, 5),
            "grad_norm": round(grad_norm, 5),
            "lr": cur_lr,
            "sup_tokens": sup_toks,
            "sec": round(dt, 1),
        })
        print(
            f"  [Step {step_counter:02d}/{total_opt_steps:02d}] loss={st_loss:.4f} (act={st_act:.4f}) | gnorm={grad_norm:.3f} | lr={cur_lr:.2e} | {dt:.1f}s",
            flush=True,
        )

        if step_counter in ckpt_steps:
            tag = f"CKPT-S{step_counter:02d}"
            snap = snapshot_lora_cpu()
            ev = evaluate_held_out(tag)
            ckpt_candidates.append((tag, ev, snap))
            export_adapter_dir(OUT_ADAPTER_DIR / "checkpoints" / tag, snap, target_module_names, scale_mult=1.0)

    # Select best checkpoint ensuring Overall NLL & Action NLL improve without Thought spike
    valid_ckpts = [c for c in ckpt_candidates if c[1]["thought_nll"] <= pre_eval["thought_nll"] + 0.015]
    pool = valid_ckpts or ckpt_candidates
    best_tag, best_raw_eval, best_snap = min(pool, key=lambda x: (x[1]["overall_nll"] + x[1]["action_nll"]))
    print(f"\n[*] Best Checkpoint: {best_tag} (Overall={best_raw_eval['overall_nll']:.5f}, Action={best_raw_eval['action_nll']:.5f})")

    # Scale Sweep on Held-Out Repos (effective alpha/r in [0.50, 0.375, 0.25, 0.125])
    scale_results = []
    for mult in [1.0, 0.75, 0.50, 0.25]:
        if mult == 1.0:
            ev_s = best_raw_eval
        else:
            restore_lora_cpu(best_snap, scale_mult=mult)
            ev_s = evaluate_held_out(f"SCALE-{mult:.2f}")
        scale_results.append((mult, ev_s))

    best_mult, best_post_eval = min(scale_results, key=lambda x: (x[1]["overall_nll"] + x[1]["action_nll"]))
    print(
        f"\n[OK] Final Selection: {best_tag} x {best_mult:.2f} (eff scale={best_mult * LORA_ALPHA / LORA_R:.4f})\n"
        f"  - Pre-Train : Overall={pre_eval['overall_nll']:.5f} | Action={pre_eval['action_nll']:.5f} | Thought={pre_eval['thought_nll']:.5f}\n"
        f"  - Post-Train: Overall={best_post_eval['overall_nll']:.5f} | Action={best_post_eval['action_nll']:.5f} | Thought={best_post_eval['thought_nll']:.5f}",
        flush=True,
    )

    export_adapter_dir(OUT_ADAPTER_DIR, best_snap, target_module_names, scale_mult=best_mult)
    sha = package_submission(OUT_ADAPTER_DIR, OUT_ZIP)
    print(f"[OK] Packaged {OUT_ZIP} ({OUT_ZIP.stat().st_size / (1024*1024):.2f} MB) | SHA256={sha}")

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps({
        "adapter_dir": str(OUT_ADAPTER_DIR),
        "submission_zip": str(OUT_ZIP),
        "sha256": sha,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "best_checkpoint": best_tag,
        "best_scale_mult": best_mult,
        "effective_scale": best_mult * (LORA_ALPHA / LORA_R),
        "target_modules_count": len(target_module_names),
        "val_repos": val_repos,
        "audit_stats": audit_stats,
        "pre_train_val": pre_eval,
        "post_train_val": best_post_eval,
        "scale_sweep": [{"scale_mult": m, **ev} for m, ev in scale_results],
        "train_history": train_history,
    }, indent=2), encoding="utf-8")

    tmp_sub = Path("/tmp/submission.zip")
    shutil.copy2(OUT_ZIP, tmp_sub)
    res = subprocess.run([sys.executable, str(ROOT / "data" / "dmitrii_kit" / "check_submission.py"), str(tmp_sub)], capture_output=True, text=True)
    print(res.stdout)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
