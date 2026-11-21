#!/usr/bin/env python3
"""
Train & Package v48: Edit-Only Upper-Layer LoRA (layers 30..59, q_proj/o_proj/down_proj = 90 modules)
on data/dmitrii_kit (native + external_A + external_B source edit_file calls only), packaged on top of v47.

Key design (informed by ayush04barai 0.22 LoRA + v46 failure analysis):
1. Loss ONLY on `<|tool_call>call:edit_file{...}` tokens on source files (0 loss on thoughts,
   0 loss on run_command/read_file/write_file -> zero thought collapse, zero repro/grep loop drift).
2. Safely mixes native Gemma 4 + Qwen3-Coder (external_A / external_B) resolved edit_file turns
   (empty thought channels in external_B are completely masked out).
3. Lightweight 90-module adapter (top 30 layers x 3 projections: q_proj, o_proj, down_proj) ->
   4.5x smaller & faster than v46's 410-module adapter, with explicit target_modules list avoiding
   vision_tower Gemma4ClippableLinear.
4. Held-out repo NLL + scale sweep [1.0, 0.75, 0.50, 0.25], packaged with v47's 40-call pinned prompt.
"""
import argparse
import hashlib
import json
import math
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
from package_v47_candidate import V47_EVAL_YAML, V47_SYSTEM_MD

SEED = 42
DATA_DIR = ROOT / "data" / "dmitrii_kit"
QAT_DIR = Path("/home/lmassaron/models/kaggle/gemma-4-31b-it-qat-w4a16-ct")
OUT_ADAPTER_DIR = ROOT / "models" / "lora_v48_edit_only"
OUT_ZIP = ROOT / "submission_v48_patchsmith_v11_edit_lora.zip"
REPORT_FILE = ROOT / "state_of_the_art" / "benchmarks_data" / "v48_edit_lora_training_report.json"

DATA_FILES = [
    "lora_native_resolved.jsonl",
    "lora_external_A.jsonl",
    "lora_external_B.part000.jsonl",
    "lora_external_B.part001.jsonl",
    "lora_external_B.part002.jsonl",
    "lora_external_B.part003.jsonl",
]

MAX_CONTEXT = 12288
MAX_TARGET = 1536
LORA_R = 16
LORA_ALPHA = 16  # scale = 1.0 max; swept over [1.0, 0.75, 0.50, 0.25] on held-out edit NLL
LORA_DROPOUT = 0.0
LAYERS = set(range(30, 60))  # top 30 layers
TARGET_MAP = {"q_proj": "self_attn", "o_proj": "self_attn", "down_proj": "mlp"}  # 90 modules
MOD_PAT = re.compile(r"^model\.language_model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+_proj)$")

GRAD_ACCUM = 4
MAX_TRAIN_EXAMPLES = 240  # 60 optimizer steps (balanced native + external source edit_file calls)
VAL_EXAMPLES = 32         # held-out repo edit_file examples
LR = 1.0e-4
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 4

EDIT_CALL_RE = re.compile(r"<\|tool_call>call:edit_file\{")
PATH_RE = re.compile(r'filepath:<\|"\|>(.*?)<\|"\|>')
NOT_SOURCE_RE = re.compile(
    r"(^|/)(tests?|testing)/|(^|/)test_[^/]*$|_test\.py$|^/?tmp/|conftest\.py$|"
    r"(^|/)(repro|reproduce|verify|debug|check|scratch|demo)[^/]*$",
    re.IGNORECASE,
)
TOOL_CALL_ID = 48

V48_AGENT_YAML = """name: patchsmith
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

V48_SAMPLING_YAML = """temperature: 0.2
top_p: 0.95
max_output_tokens: 6144
thinking_config:
  thinking_budget: 4096
  include_thoughts: true
"""


def curate_edit_only_dataset():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("google/gemma-4-31B-it")
    assert tok.convert_tokens_to_ids("<|tool_call>") == TOOL_CALL_ID

    stats = Counter()
    examples = []

    for fname in DATA_FILES:
        fpath = DATA_DIR / fname
        if not fpath.is_file():
            continue
        is_native = "native" in fname
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                repo = (rec.get("repository") or rec.get("repo") or "unknown").lower()
                exs = rec.get("sft_examples") or []
                for idx, ex in enumerate(exs):
                    stats["total_turns"] += 1
                    text = ex.get("completion") or ""
                    if text.count("<|tool_call>") != 1 or not EDIT_CALL_RE.search(text):
                        stats["skip_not_edit"] += 1
                        continue
                    pm = PATH_RE.search(text)
                    if not pm or NOT_SOURCE_RE.search(pm.group(1)):
                        stats["skip_non_source"] += 1
                        continue
                    # For native runs, reject edits that errored on next turn
                    if is_native and idx + 1 < len(exs):
                        next_resp = ""
                        for m in reversed(exs[idx + 1].get("request", {}).get("messages", [])):
                            if m.get("role") == "tool":
                                next_resp = str(m.get("content", ""))
                                break
                        if any(k in next_resp.lower() for k in ('"status": "error"', "not found", "no match", "multiple matches", "error:")):
                            stats["skip_native_edit_err"] += 1
                            continue
                    c_ids = ex.get("completion_token_ids")
                    if not c_ids or TOOL_CALL_ID not in c_ids or len(c_ids) > MAX_TARGET:
                        stats["skip_bad_target"] += 1
                        continue
                    ctx_toks = ex.get("context_tokens", 0)
                    if ctx_toks > MAX_CONTEXT:
                        stats["skip_over_ctx"] += 1
                        continue
                    p_ids = ex.get("prompt_token_ids")
                    if not p_ids:
                        p_ids = tok(ex["prefix"], add_special_tokens=False)["input_ids"]
                    if len(p_ids) + len(c_ids) > MAX_CONTEXT:
                        stats["skip_over_ctx"] += 1
                        continue

                    edit_start = len(p_ids) + c_ids.index(TOOL_CALL_ID)
                    edit_len = len(p_ids) + len(c_ids) - edit_start
                    if edit_len < 8:
                        stats["skip_tiny_edit"] += 1
                        continue

                    stats["kept_native" if is_native else "kept_external"] += 1
                    examples.append({
                        "repo": repo,
                        "native": is_native,
                        "ids": list(p_ids) + list(c_ids),
                        "start": edit_start,
                        "edit_tokens": edit_len,
                        "total_tokens": len(p_ids) + len(c_ids),
                    })

    rng = random.Random(SEED)
    # Exclude eval benchmark repos defensively
    eval_repos = {"psf/requests", "fastapi/fastapi", "textualize/rich", "sympy/sympy"}
    examples = [e for e in examples if e["repo"] not in eval_repos]

    # Repo-level held-out split (ensure mix of native + external in val)
    nat_repos = sorted({e["repo"] for e in examples if e["native"]})
    ext_repos = sorted({e["repo"] for e in examples if not e["native"]} - set(nat_repos))
    rng.shuffle(nat_repos)
    rng.shuffle(ext_repos)

    val_repos = set(nat_repos[: max(4, len(nat_repos) // 6)]) | set(ext_repos[: max(12, len(ext_repos) // 10)])
    val_pool = [e for e in examples if e["repo"] in val_repos]
    train_pool = [e for e in examples if e["repo"] not in val_repos]

    rng.shuffle(val_pool)
    val_nat = [e for e in val_pool if e["native"]][: VAL_EXAMPLES // 2]
    val_ext = [e for e in val_pool if not e["native"]][: VAL_EXAMPLES - len(val_nat)]
    val_selected = val_nat + val_ext

    # Interleave native Gemma edits and external Qwen3-Coder edits (repo-diverse)
    tr_nat = [e for e in train_pool if e["native"]]
    tr_ext = [e for e in train_pool if not e["native"]]
    rng.shuffle(tr_nat)
    rng.shuffle(tr_ext)

    # Deduplicate external repos round-robin so no single repo dominates
    ext_by_repo = defaultdict(list)
    for e in tr_ext:
        ext_by_repo[e["repo"]].append(e)
    ext_rkeys = sorted(ext_by_repo.keys())
    rng.shuffle(ext_rkeys)
    tr_ext_balanced = []
    ri = 0
    while len(tr_ext_balanced) < MAX_TRAIN_EXAMPLES and ext_rkeys:
        prog = False
        for rk in ext_rkeys:
            if ri < len(ext_by_repo[rk]):
                tr_ext_balanced.append(ext_by_repo[rk][ri])
                prog = True
                if len(tr_ext_balanced) >= MAX_TRAIN_EXAMPLES:
                    break
        if not prog:
            break
        ri += 1

    # Mix 1:2 native:external (re-cycling native if needed up to 2x)
    train_selected = []
    ni, ei = 0, 0
    while len(train_selected) < MAX_TRAIN_EXAMPLES and (ni < len(tr_nat) * 2 or ei < len(tr_ext_balanced)):
        if tr_nat and ni < len(tr_nat) * 2:
            train_selected.append(tr_nat[ni % len(tr_nat)])
            ni += 1
        for _ in range(2):
            if len(train_selected) < MAX_TRAIN_EXAMPLES and ei < len(tr_ext_balanced):
                train_selected.append(tr_ext_balanced[ei])
                ei += 1

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
    def __init__(self, linear: nn.Module, r: int = 0, alpha: int = 16, dropout: float = 0.0):
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


def edit_only_softcapped_loss(model: nn.Module, hidden: torch.Tensor, ids: torch.Tensor, start: int, chunk_size: int = 256):
    w = model.get_output_embeddings().weight
    out_dev = w.device
    text_cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    cap = float(getattr(text_cfg, "final_logit_softcapping", 30.0) or 30.0)

    h = hidden[0, start - 1 : -1].to(out_dev)
    y = ids[0, start:].to(out_dev)
    n_tok = int(y.numel())
    if n_tok == 0:
        return h.sum() * 0.0, 0, 0.0

    def _chunk_ce(hc: torch.Tensor, yc: torch.Tensor) -> torch.Tensor:
        logits = (hc @ w.t()).float()
        if cap > 0.0:
            logits = torch.tanh(logits / cap) * cap
        return F.cross_entropy(logits, yc, reduction="sum")

    tot = torch.zeros((), device=out_dev, dtype=torch.float32)
    for i in range(0, n_tok, chunk_size):
        hc, yc = h[i : i + chunk_size], y[i : i + chunk_size]
        loss_c = torch_checkpoint(_chunk_ce, hc, yc, use_reentrant=False) if (torch.is_grad_enabled() and h.requires_grad) else _chunk_ce(hc, yc)
        tot = tot + loss_c
    return tot / float(n_tok), n_tok, float(tot.detach().cpu())


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
            ("agent.yaml", V48_AGENT_YAML),
            ("configs/sampling.yaml", V48_SAMPLING_YAML),
            ("eval_config.yaml", V47_EVAL_YAML),
            ("prompts/system.md", V47_SYSTEM_MD),
        ]):
            info = zipfile.ZipInfo(rel, (1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            zf.writestr(info, content.encode("utf-8"))
        zf.write(adapter_dir / "adapter_config.json", arcname="adapters/main_lora/adapter_config.json")
        zf.write(adapter_dir / "adapter_model.safetensors", arcname="adapters/main_lora/adapter_model.safetensors")
    return hashlib.sha256(out_zip.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Audit edit-only dataset without GPU training")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_ex, val_ex, stats, val_repos = curate_edit_only_dataset()
    n_tr_nat = sum(1 for e in train_ex if e["native"])
    n_tr_ext = len(train_ex) - n_tr_nat
    n_vl_nat = sum(1 for e in val_ex if e["native"])
    print(f"[Audit] Stats: {stats}")
    print(f"  - Train edit examples: {len(train_ex)} ({n_tr_nat} native Gemma, {n_tr_ext} external Qwen) across {len({e['repo'] for e in train_ex})} repos")
    print(f"  - Val edit examples  : {len(val_ex)} ({n_vl_nat} native, {len(val_ex)-n_vl_nat} external) across {len(val_repos)} held-out repos")
    print(f"  - Edit tokens train  : {sum(e['edit_tokens'] for e in train_ex):,} | max ctx: {max(e['total_tokens'] for e in train_ex)}")

    if args.dry_run:
        return

    import transformers
    from transformers import AutoConfig

    t0 = time.time()
    print(f"\n[*] Loading {QAT_DIR} in packed INT4 on GPU (top-30-layer 90-module LoRA)...")
    cfg = AutoConfig.from_pretrained(str(QAT_DIR), trust_remote_code=True)
    cls = getattr(transformers, cfg.architectures[0])
    base_model = cls.from_pretrained(str(QAT_DIR), dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    if hasattr(base_model, "ct_decompress_hook"):
        base_model.ct_decompress_hook.remove()
        del base_model.ct_decompress_hook

    for p in base_model.parameters():
        p.requires_grad_(False)

    injected = {}
    target_names = []
    swapped = 0
    for name, mod in list(base_model.named_modules()):
        if not hasattr(mod, "weight_packed"):
            continue
        child = name.rsplit(".", 1)[-1]
        m = MOD_PAT.match(name)
        use_lora = bool(m and int(m.group(1)) in LAYERS and TARGET_MAP.get(m.group(3)) == m.group(2))
        parent = name.rsplit(".", 1)[0]
        lm = Int4LoRALinear(mod, r=LORA_R if use_lora else 0, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
        setattr(base_model.get_submodule(parent), child, lm)
        swapped += 1
        if use_lora:
            injected[name] = lm
            target_names.append(name)

    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base_model.config.use_cache = False
    if hasattr(base_model.config, "text_config") and base_model.config.text_config is not None:
        base_model.config.text_config.use_cache = False

    trainable = [p for p in base_model.parameters() if p.requires_grad]
    print(f"[OK] Loaded in {time.time()-t0:.1f}s | Swapped {swapped} INT4 | Injected {len(injected)} LoRA modules ({sum(p.numel() for p in trainable):,} params)", flush=True)
    first_dev = base_model.get_input_embeddings().weight.device

    def snap_cpu():
        return {n: (m.lora_A.detach().cpu().clone(), m.lora_B.detach().cpu().clone()) for n, m in injected.items()}

    def restore_cpu(snap, mult: float = 1.0):
        with torch.no_grad():
            for n, m in injected.items():
                A_c, B_c = snap[n]
                m.lora_A.copy_(A_c.to(m.lora_A.device, dtype=m.lora_A.dtype))
                m.lora_B.copy_((B_c * mult).to(m.lora_B.device, dtype=m.lora_B.dtype))

    def eval_val(tag: str) -> dict:
        tv = time.time()
        base_model.eval()
        tot_nll, tot_tok = 0.0, 0
        with torch.no_grad():
            for ex in val_ex:
                ids = torch.tensor([ex["ids"]], dtype=torch.long, device=first_dev)
                hid = base_model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
                _, nt, s_nll = edit_only_softcapped_loss(base_model, hid, ids, ex["start"])
                tot_nll += s_nll
                tot_tok += nt
        base_model.train()
        nll = tot_nll / max(1, tot_tok)
        print(f"  [VAL {tag:14s}] Edit-Span NLL={nll:.5f} ({tot_tok} tok) [{time.time()-tv:.1f}s]", flush=True)
        return {"edit_nll": nll, "edit_tokens": tot_tok}

    pre_val = eval_val("PRE-TRAIN")
    opt = torch.optim.AdamW(trainable, lr=LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
    total_opt = len(train_ex) // GRAD_ACCUM
    ckpt_steps = {20, 40, total_opt}
    ckpts = []
    history = []

    def lr_at(idx: int) -> float:
        if idx < WARMUP_STEPS:
            return LR * float(idx + 1) / float(WARMUP_STEPS)
        prog = min(1.0, float(idx - WARMUP_STEPS) / max(1.0, float(total_opt - WARMUP_STEPS)))
        return LR * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))

    print(f"\n[*] Training {total_opt} opt steps ({len(train_ex)} edit_file examples, accum={GRAD_ACCUM}, lr={LR})...", flush=True)
    step = 0
    for s in range(0, len(train_ex), GRAD_ACCUM):
        batch = train_ex[s : s + GRAD_ACCUM]
        cur_lr = lr_at(step)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        ts = time.time()
        b_nll, b_tok = 0.0, 0
        for ex in batch:
            ids = torch.tensor([ex["ids"]], dtype=torch.long, device=first_dev)
            hid = base_model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
            loss_v, nt, s_nll = edit_only_softcapped_loss(base_model, hid, ids, ex["start"])
            if nt > 0:
                (loss_v / float(len(batch))).backward()
                b_nll += s_nll
                b_tok += nt
        gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).cpu())
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        st_nll = b_nll / max(1, b_tok)
        dt = time.time() - ts
        history.append({"step": step, "edit_nll": round(st_nll, 5), "gnorm": round(gnorm, 4), "lr": cur_lr, "sec": round(dt, 1)})
        print(f"  [Step {step:02d}/{total_opt:02d}] edit_nll={st_nll:.4f} | gnorm={gnorm:.3f} | lr={cur_lr:.2e} | {dt:.1f}s", flush=True)
        if step in ckpt_steps:
            tag = f"CKPT-S{step:02d}"
            sn = snap_cpu()
            ev = eval_val(tag)
            ckpts.append((tag, ev, sn))
            export_adapter_dir(OUT_ADAPTER_DIR / "checkpoints" / tag, sn, target_names, scale_mult=1.0)

    best_tag, best_raw, best_snap = min(ckpts, key=lambda x: x[1]["edit_nll"])
    scales = []
    for mult in [1.0, 0.75, 0.50, 0.25]:
        if mult == 1.0:
            ev_s = best_raw
        else:
            restore_cpu(best_snap, mult=mult)
            ev_s = eval_val(f"SCALE-{mult:.2f}")
        scales.append((mult, ev_s))

    best_mult, best_post = min(scales, key=lambda x: x[1]["edit_nll"])
    print(f"\n[OK] Selected {best_tag} x {best_mult:.2f} (eff scale={best_mult * LORA_ALPHA / LORA_R:.4f}): {pre_val['edit_nll']:.5f} -> {best_post['edit_nll']:.5f}", flush=True)
    export_adapter_dir(OUT_ADAPTER_DIR, best_snap, target_names, scale_mult=best_mult)
    sha = package_submission(OUT_ADAPTER_DIR, OUT_ZIP)
    print(f"[OK] Packaged {OUT_ZIP} ({OUT_ZIP.stat().st_size/(1024*1024):.2f} MB) | SHA256={sha}")

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps({
        "adapter_dir": str(OUT_ADAPTER_DIR),
        "submission_zip": str(OUT_ZIP),
        "sha256": sha,
        "modules": len(target_names),
        "best_checkpoint": best_tag,
        "best_scale_mult": best_mult,
        "effective_scale": best_mult * LORA_ALPHA / LORA_R,
        "pre_train_val": pre_val,
        "post_train_val": best_post,
        "scale_sweep": [{"scale_mult": m, **ev} for m, ev in scales],
        "audit_stats": stats,
        "train_history": history,
    }, indent=2), encoding="utf-8")

    tmp_sub = Path("/tmp/submission.zip")
    shutil.copy2(OUT_ZIP, tmp_sub)
    res = subprocess.run([sys.executable, str(ROOT / "data" / "dmitrii_kit" / "check_submission.py"), str(tmp_sub)], capture_output=True, text=True)
    print(res.stdout)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
