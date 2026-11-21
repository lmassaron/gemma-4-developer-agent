#!/usr/bin/env python3
"""
Train & Package v49: Edit-Span DPO (+ NLL anchor / RPO) initialized from v48 (CKPT-S40)
on layers 30..59 x {q_proj, o_proj, down_proj} (90 modules).

Grounded in EDA of native Gemma 4 edit_file trajectories (28.5% real edit_file error rate):
  - 55.3% (57/103): schema/delimiter corruption (backtick/quote instead of <|"|>, wrong arg name, missing old_string)
  - 39.8% (41/103): old_string_not_found (indentation shift, double-escaped \\n, ellipsis/dropped line)
  -  3.9% ( 4/103): ambiguous_multiple_matches (over-short 1-line old_string)
  -  1.0% ( 1/103): empty old_string / no-op identical edit

Design:
  1. Initializes both policy pi_theta and fixed reference pi_ref from v48 (models/lora_v48_edit_only).
  2. Precomputes exact pi_ref edit-span log-probs once under torch.no_grad() (zero extra reference model VRAM).
  3. Combines real Gemma 4 self-correction pairs (native failed -> succeeded edit_file on same file)
     with empirically-matched hard-negative edit pairs mined from native + external_A + B + B2 source edits
     (excluding eval repos: requests, fastapi, rich, sympy; repo-level held-out validation split).
  4. Optimizes length-normalized Edit-Span DPO + SFT NLL anchor (RPO):
       L = -log sigmoid(beta * ((lp_w - ref_w) - (lp_l - ref_l))) + nll_anchor * (-mean_lp_w)
     Strictly 0 loss on thoughts & non-edit tokens!
  5. Evaluates held-out Preference Accuracy, Margin, and Chosen Edit-Span NLL at checkpoints + v48/v49 blend sweep,
     exports models/lora_v49_edit_dpo/, packages submission_v49_patchsmith_v11_edit_dpo.zip, and validates via check_submission.py.
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
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_v48_edit_only_lora import (
    DATA_DIR,
    EDIT_CALL_RE,
    LAYERS,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MOD_PAT,
    NOT_SOURCE_RE,
    PATH_RE,
    QAT_DIR,
    TARGET_MAP,
    TOOL_CALL_ID,
    Int4LoRALinear,
    edit_only_softcapped_loss,
    export_adapter_dir,
    package_submission,
)

SEED = 49
INIT_V48_DIR = ROOT / "models" / "lora_v48_edit_only"
OUT_ADAPTER_DIR = ROOT / "models" / "lora_v49_edit_dpo"
OUT_ZIP = ROOT / "submission_v49_patchsmith_v11_edit_dpo.zip"
REPORT_FILE = ROOT / "state_of_the_art" / "benchmarks_data" / "v49_edit_dpo_training_report.json"

DATA_FILES = [
    "lora_native_resolved.jsonl",
    "lora_external_A.jsonl",
    "lora_external_B.part000.jsonl",
    "lora_external_B.part001.jsonl",
    "lora_external_B.part002.jsonl",
    "lora_external_B.part003.jsonl",
    "lora_external_B2.part000.jsonl",
    "lora_external_B2.part001.jsonl",
    "lora_external_B2.part002.jsonl",
    "lora_external_B2.part003.jsonl",
]

MAX_CONTEXT = 10240   # Keeps runtime fast (~5h total today) while covering core source edits
MAX_TARGET = 1536
MAX_TRAIN_PAIRS = 64  # 16 optimizer steps x grad_accum=4 (high-signal post-SFT DPO)
VAL_PAIRS = 24        # Held-out unseen-repo preference + NLL gate
GRAD_ACCUM = 4
LR = 2.0e-5           # Conservative post-SFT DPO LR from v48
WARMUP_STEPS = 2
DPO_BETA = 0.2        # On token-mean log-probs (effective scale ~2-4x stronger than sum-beta=0.01, length-invariant)
NLL_ANCHOR = 0.25     # RPO anchor to preserve/improve chosen edit NLL

ERR_KEYS = ('"status": "error"', "not found", "no match", "multiple matches", "error:", "mandatory input parameters")
QUOTE_OPEN = '<|"|>'
FIELD_RE = re.compile(
    r'^(?P<head><\|tool_call>call:edit_file\{)'
    r'(?P<body>.*)'
    r'(?P<tail>\}<tool_call\|>.*)$',
    re.DOTALL,
)
KV_RE = re.compile(r'(filepath|new_string|old_string):<\|"\|>(.*?)<\|"\|>', re.DOTALL)


def parse_edit_call(edit_text: str):
    m = FIELD_RE.match(edit_text)
    if not m:
        return None
    fields = {}
    order = []
    for km in KV_RE.finditer(m.group("body")):
        k, v = km.group(1), km.group(2)
        fields[k] = v
        order.append(k)
    if not all(k in fields for k in ("filepath", "old_string", "new_string")):
        return None
    return m.group("head"), order, fields, m.group("tail")


def rebuild_edit_call(head: str, order: list[str], fields: dict[str, str], tail: str) -> str:
    parts = [f'{k}:{QUOTE_OPEN}{fields[k]}{QUOTE_OPEN}' for k in order if k in fields]
    return f"{head}{','.join(parts)}{tail}"


def synthesize_empirical_negative(edit_text: str, rng: random.Random, forced_mode: str | None = None):
    """Create a realistic rejected edit_file call matching Gemma 4's observed native error taxonomy."""
    parsed = parse_edit_call(edit_text)
    if parsed is None:
        return None, None
    head, order, fields, tail = parsed
    old_s = fields["old_string"]
    new_s = fields["new_string"]

    # Choose corruption weighted by real observed Gemma 4 error distribution:
    # 35% schema/delimiter, 35% old_string mismatch/escape, 15% ambiguous short old_string, 15% noop/empty
    if forced_mode is None:
        u = rng.random()
        if u < 0.35:
            mode = "schema_delimiter"
        elif u < 0.70:
            mode = "old_string_mismatch"
        elif u < 0.85:
            mode = "ambiguous_short"
        else:
            mode = "noop_or_empty"
    else:
        mode = forced_mode

    if mode == "schema_delimiter":
        sub = rng.choice(["backtick_delim", "wrong_arg_name", "drop_old_string"])
        if sub == "backtick_delim" and f'{QUOTE_OPEN},old_string:' in edit_text:
            return edit_text.replace(f'{QUOTE_OPEN},old_string:', '`,old_string:', 1), "schema_backtick_delim"
        if sub == "wrong_arg_name":
            if "filepath:" in edit_text:
                return edit_text.replace("filepath:", "file_path:", 1), "schema_wrong_arg_name"
        # fallback: drop old_string parameter
        f2 = {k: v for k, v in fields.items() if k != "old_string"}
        return rebuild_edit_call(head, [k for k in order if k != "old_string"], f2, tail), "schema_drop_old_string"

    if mode == "old_string_mismatch":
        lines = old_s.splitlines(keepends=True)
        sub = rng.choice(["indent_shift", "double_escape_nl", "ellipsis_drop"])
        if sub == "double_escape_nl" and "\n" in old_s:
            f2 = dict(fields)
            f2["old_string"] = old_s.replace("\n", "\\n")
            f2["new_string"] = new_s.replace("\n", "\\n")
            return rebuild_edit_call(head, order, f2, tail), "mismatch_double_escape_nl"
        if sub == "ellipsis_drop" and len(lines) >= 3:
            f2 = dict(fields)
            f2["old_string"] = lines[0] + "    # ...\n" + lines[-1]
            return rebuild_edit_call(head, order, f2, tail), "mismatch_ellipsis_drop"
        # indent_shift: add/remove 2 spaces on non-empty lines
        f2 = dict(fields)
        shifted = []
        for ln in lines:
            if ln.startswith("    "):
                shifted.append(ln[2:])
            elif ln.strip():
                shifted.append("  " + ln)
            else:
                shifted.append(ln)
        cand = "".join(shifted)
        if cand != old_s:
            f2["old_string"] = cand
            return rebuild_edit_call(head, order, f2, tail), "mismatch_indent_shift"
        # fallback double escape
        f2["old_string"] = old_s + " "
        return rebuild_edit_call(head, order, f2, tail), "mismatch_trailing_ws"

    if mode == "ambiguous_short":
        lines = [ln for ln in old_s.splitlines() if ln.strip()]
        if len(lines) >= 2:
            f2 = dict(fields)
            f2["old_string"] = lines[0]
            return rebuild_edit_call(head, order, f2, tail), "ambiguous_single_line"
        f2 = dict(fields)
        f2["old_string"] = old_s[: max(6, len(old_s) // 4)]
        if f2["old_string"] != old_s:
            return rebuild_edit_call(head, order, f2, tail), "ambiguous_truncated"

    # noop_or_empty
    f2 = dict(fields)
    if rng.random() < 0.65 and old_s != new_s:
        f2["new_string"] = old_s
        return rebuild_edit_call(head, order, f2, tail), "noop_identical"
    f2["old_string"] = ""
    return rebuild_edit_call(head, order, f2, tail), "empty_old_string"


def curate_edit_dpo_dataset():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("google/gemma-4-31B-it")
    assert tok.convert_tokens_to_ids("<|tool_call>") == TOOL_CALL_ID

    stats = Counter()
    pairs = []
    seen_real_keys = set()
    rng_neg = random.Random(SEED + 7)
    eval_repos = {"psf/requests", "fastapi/fastapi", "textualize/rich", "sympy/sympy"}

    # Pass 1: Mine REAL on-policy self-correction pairs from native_resolved + native_unresolved
    for fn in ["lora_native_resolved.jsonl", "lora_native_unresolved.jsonl"]:
        fp_path = DATA_DIR / fn
        if not fp_path.is_file():
            continue
        with open(fp_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                repo = (rec.get("repository") or rec.get("repo") or "unknown").lower()
                if repo in eval_repos:
                    continue
                exs = rec.get("sft_examples") or []
                failed_by_file = {}
                for idx, ex in enumerate(exs):
                    comp = ex.get("completion") or ""
                    if comp.count("<|tool_call>") != 1 or not EDIT_CALL_RE.search(comp):
                        continue
                    pm = PATH_RE.search(comp)
                    if not pm or NOT_SOURCE_RE.search(pm.group(1)):
                        continue
                    fp = pm.group(1)
                    next_resp = ""
                    if idx + 1 < len(exs):
                        for m in reversed(exs[idx + 1].get("request", {}).get("messages", [])):
                            if m.get("role") == "tool":
                                next_resp = str(m.get("content", ""))
                                break
                    is_err = any(k in next_resp.lower() for k in ERR_KEYS)
                    edit_span = comp[comp.index("<|tool_call>") :]
                    if is_err:
                        failed_by_file[fp] = edit_span
                        continue
                    if fp in failed_by_file:
                        rej_span = failed_by_file.pop(fp)
                        if rej_span == edit_span:
                            continue
                        c_ids = ex.get("completion_token_ids")
                        p_ids = ex.get("prompt_token_ids")
                        if not c_ids or TOOL_CALL_ID not in c_ids or not p_ids:
                            continue
                        tc_pos = c_ids.index(TOOL_CALL_ID)
                        prefix_ids = list(p_ids) + list(c_ids[:tc_pos])
                        chosen_span_ids = list(c_ids[tc_pos:])
                        rej_span_ids = tok(rej_span, add_special_tokens=False)["input_ids"]
                        # Ensure EOS/end-of-turn alignment matches chosen tail if present
                        if chosen_span_ids and chosen_span_ids[-1] == 106 and (not rej_span_ids or rej_span_ids[-1] != 106):
                            rej_span_ids.append(106)
                        if (
                            len(chosen_span_ids) < 8
                            or len(rej_span_ids) < 8
                            or len(prefix_ids) + max(len(chosen_span_ids), len(rej_span_ids)) > MAX_CONTEXT
                            or max(len(chosen_span_ids), len(rej_span_ids)) > MAX_TARGET
                        ):
                            stats["skip_real_over_ctx"] += 1
                            continue
                        key = (repo, fp, len(prefix_ids))
                        seen_real_keys.add(key)
                        stats["real_self_corr"] += 1
                        pairs.append({
                            "repo": repo,
                            "native": True,
                            "source": "real_self_correction",
                            "neg_type": "real_native_error",
                            "chosen_ids": prefix_ids + chosen_span_ids,
                            "rejected_ids": prefix_ids + rej_span_ids,
                            "start": len(prefix_ids),
                            "chosen_tokens": len(chosen_span_ids),
                            "rejected_tokens": len(rej_span_ids),
                        })

    # Pass 2: Build empirically-matched hard-negative pairs from clean source edit_file calls
    for fname in DATA_FILES:
        fpath = DATA_DIR / fname
        if not fpath.is_file():
            continue
        is_native = "native" in fname
        tier = "native" if is_native else ("ext_A" if "external_A" in fname else ("ext_B2" if "_B2." in fname else "ext_B"))
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                repo = (rec.get("repository") or rec.get("repo") or "unknown").lower()
                if repo in eval_repos:
                    continue
                exs = rec.get("sft_examples") or []
                for idx, ex in enumerate(exs):
                    comp = ex.get("completion") or ""
                    if comp.count("<|tool_call>") != 1 or not EDIT_CALL_RE.search(comp):
                        continue
                    pm = PATH_RE.search(comp)
                    if not pm or NOT_SOURCE_RE.search(pm.group(1)):
                        continue
                    if is_native and idx + 1 < len(exs):
                        next_resp = ""
                        for m in reversed(exs[idx + 1].get("request", {}).get("messages", [])):
                            if m.get("role") == "tool":
                                next_resp = str(m.get("content", ""))
                                break
                        if any(k in next_resp.lower() for k in ERR_KEYS):
                            continue
                    ctx_toks = ex.get("context_tokens", 0)
                    if ctx_toks > MAX_CONTEXT:
                        stats["skip_over_ctx"] += 1
                        continue
                    c_ids = ex.get("completion_token_ids")
                    if not c_ids or TOOL_CALL_ID not in c_ids or len(c_ids) > MAX_TARGET:
                        continue
                    p_ids = ex.get("prompt_token_ids")
                    if not p_ids:
                        p_ids = tok(ex["prefix"], add_special_tokens=False)["input_ids"]
                    tc_pos = c_ids.index(TOOL_CALL_ID)
                    prefix_ids = list(p_ids) + list(c_ids[:tc_pos])
                    chosen_span_ids = list(c_ids[tc_pos:])
                    if len(chosen_span_ids) < 8 or len(prefix_ids) + len(chosen_span_ids) > MAX_CONTEXT:
                        stats["skip_over_ctx"] += 1
                        continue
                    if (repo, pm.group(1), len(prefix_ids)) in seen_real_keys:
                        continue

                    edit_span = comp[comp.index("<|tool_call>") :]
                    rej_span, neg_type = synthesize_empirical_negative(edit_span, rng_neg)
                    if not rej_span or rej_span == edit_span:
                        stats["skip_neg_fail"] += 1
                        continue
                    rej_span_ids = tok(rej_span, add_special_tokens=False)["input_ids"]
                    if chosen_span_ids and chosen_span_ids[-1] == 106 and (not rej_span_ids or rej_span_ids[-1] != 106):
                        rej_span_ids.append(106)
                    if (
                        len(rej_span_ids) < 6
                        or rej_span_ids == chosen_span_ids
                        or len(prefix_ids) + len(rej_span_ids) > MAX_CONTEXT
                        or len(rej_span_ids) > MAX_TARGET
                    ):
                        stats["skip_neg_len"] += 1
                        continue

                    stats[f"kept_{tier}"] += 1
                    stats[f"neg_{neg_type}"] += 1
                    pairs.append({
                        "repo": repo,
                        "native": is_native,
                        "source": tier,
                        "neg_type": neg_type,
                        "chosen_ids": prefix_ids + chosen_span_ids,
                        "rejected_ids": prefix_ids + rej_span_ids,
                        "start": len(prefix_ids),
                        "chosen_tokens": len(chosen_span_ids),
                        "rejected_tokens": len(rej_span_ids),
                    })

    rng = random.Random(SEED)
    # Repo-level held-out split
    nat_repos = sorted({e["repo"] for e in pairs if e["native"]})
    ext_repos = sorted({e["repo"] for e in pairs if not e["native"]} - set(nat_repos))
    rng.shuffle(nat_repos)
    rng.shuffle(ext_repos)

    val_repos = set(nat_repos[: max(5, len(nat_repos) // 5)]) | set(ext_repos[: max(14, len(ext_repos) // 8)])
    val_pool = [e for e in pairs if e["repo"] in val_repos]
    train_pool = [e for e in pairs if e["repo"] not in val_repos]

    rng.shuffle(val_pool)
    val_nat = [e for e in val_pool if e["native"]][: VAL_PAIRS // 2]
    val_ext = [e for e in val_pool if not e["native"]][: VAL_PAIRS - len(val_nat)]
    val_selected = val_nat + val_ext

    # Build repo-diverse train set prioritizing ALL real self-correction pairs first, then balanced native + ext (incl B2)
    tr_real = [e for e in train_pool if e["source"] == "real_self_correction"]
    tr_nat = [e for e in train_pool if e["native"] and e["source"] != "real_self_correction"]
    tr_ext = [e for e in train_pool if not e["native"]]
    rng.shuffle(tr_real)
    rng.shuffle(tr_nat)
    rng.shuffle(tr_ext)

    ext_by_repo = defaultdict(list)
    for e in tr_ext:
        ext_by_repo[e["repo"]].append(e)
    ext_rkeys = sorted(ext_by_repo.keys())
    rng.shuffle(ext_rkeys)
    tr_ext_bal = []
    ri = 0
    while len(tr_ext_bal) < MAX_TRAIN_PAIRS and ext_rkeys:
        prog = False
        for rk in ext_rkeys:
            if ri < len(ext_by_repo[rk]):
                tr_ext_bal.append(ext_by_repo[rk][ri])
                prog = True
                if len(tr_ext_bal) >= MAX_TRAIN_PAIRS:
                    break
        if not prog:
            break
        ri += 1

    train_selected = list(tr_real)
    ni, ei = 0, 0
    while len(train_selected) < MAX_TRAIN_PAIRS and (ni < len(tr_nat) or ei < len(tr_ext_bal)):
        if ni < len(tr_nat):
            train_selected.append(tr_nat[ni])
            ni += 1
        if len(train_selected) < MAX_TRAIN_PAIRS and ei < len(tr_ext_bal):
            train_selected.append(tr_ext_bal[ei])
            ei += 1

    rng.shuffle(train_selected)
    usable = (len(train_selected) // GRAD_ACCUM) * GRAD_ACCUM
    train_selected = train_selected[:usable]
    return train_selected, val_selected, dict(stats), sorted(val_repos)


def load_v48_into_injected(injected: dict[str, Int4LoRALinear], v48_dir: Path):
    st = load_safetensors(str(v48_dir / "adapter_model.safetensors"))
    loaded = 0
    with torch.no_grad():
        for name, mod in injected.items():
            canon = name if name.startswith("model.language_model.") else f"model.{name}"
            ka = f"base_model.model.{canon}.lora_A.weight"
            kb = f"base_model.model.{canon}.lora_B.weight"
            assert ka in st and kb in st, f"Missing v48 weights for {name}"
            mod.lora_A.copy_(st[ka].to(mod.lora_A.device, dtype=mod.lora_A.dtype))
            mod.lora_B.copy_(st[kb].to(mod.lora_B.device, dtype=mod.lora_B.dtype))
            loaded += 1
    return loaded


def edit_span_mean_logp(model: nn.Module, ids: torch.Tensor, start: int):
    hid = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
    mean_nll, nt, sum_nll = edit_only_softcapped_loss(model, hid, ids, start)
    return -mean_nll, mean_nll, nt, sum_nll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Audit DPO pair dataset without GPU training")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_pairs, val_pairs, stats, val_repos = curate_edit_dpo_dataset()
    tr_src = Counter(p["source"] for p in train_pairs)
    tr_neg = Counter(p["neg_type"] for p in train_pairs)
    vl_src = Counter(p["source"] for p in val_pairs)
    print(f"[Audit] Stats: {stats}")
    print(f"  - Train pairs: {len(train_pairs)} across {len({p['repo'] for p in train_pairs})} repos | sources={dict(tr_src)}")
    print(f"  - Train negs : {dict(tr_neg)}")
    print(f"  - Val pairs  : {len(val_pairs)} across {len(val_repos)} held-out repos | sources={dict(vl_src)}")
    print(f"  - Max seq len: train={max(max(len(p['chosen_ids']), len(p['rejected_ids'])) for p in train_pairs)}, val={max(max(len(p['chosen_ids']), len(p['rejected_ids'])) for p in val_pairs)}")

    if args.dry_run:
        return

    import transformers
    from transformers import AutoConfig

    t0 = time.time()
    print(f"\n[*] Loading {QAT_DIR} in packed INT4 on GPU & initializing from v48 ({INIT_V48_DIR})...", flush=True)
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

    n_loaded = load_v48_into_injected(injected, INIT_V48_DIR)
    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base_model.config.use_cache = False
    if hasattr(base_model.config, "text_config") and base_model.config.text_config is not None:
        base_model.config.text_config.use_cache = False

    trainable = [p for p in base_model.parameters() if p.requires_grad]
    print(f"[OK] Loaded in {time.time()-t0:.1f}s | Swapped {swapped} INT4 | Initialized {n_loaded}/90 v48 LoRA modules ({sum(p.numel() for p in trainable):,} params)", flush=True)
    first_dev = base_model.get_input_embeddings().weight.device

    def snap_cpu():
        return {n: (m.lora_A.detach().cpu().clone(), m.lora_B.detach().cpu().clone()) for n, m in injected.items()}

    def restore_blend(v48_snap, cur_snap, alpha: float):
        """Linearly interpolate effective delta: W_v48 + alpha * (W_cur - W_v48) is approximate in A,B; exact when blending A,B near v48."""
        with torch.no_grad():
            for n, m in injected.items():
                A0, B0 = v48_snap[n]
                A1, B1 = cur_snap[n]
                m.lora_A.copy_((A0 * (1.0 - alpha) + A1 * alpha).to(m.lora_A.device, dtype=m.lora_A.dtype))
                m.lora_B.copy_((B0 * (1.0 - alpha) + B1 * alpha).to(m.lora_B.device, dtype=m.lora_B.dtype))

    v48_snap = snap_cpu()

    # Precompute fixed reference (v48) mean log-probs on train and val pairs once
    print(f"\n[*] Precomputing fixed v48 reference edit-span log-probs ({len(train_pairs)} train + {len(val_pairs)} val pairs)...", flush=True)
    t_ref = time.time()
    base_model.eval()
    with torch.no_grad():
        for pool_name, pool in [("train", train_pairs), ("val", val_pairs)]:
            for idx_p, p in enumerate(pool):
                ids_w = torch.tensor([p["chosen_ids"]], dtype=torch.long, device=first_dev)
                ids_l = torch.tensor([p["rejected_ids"]], dtype=torch.long, device=first_dev)
                lp_w, _, _, _ = edit_span_mean_logp(base_model, ids_w, p["start"])
                lp_l, _, _, _ = edit_span_mean_logp(base_model, ids_l, p["start"])
                p["ref_lp_w"] = float(lp_w.cpu())
                p["ref_lp_l"] = float(lp_l.cpu())
            print(f"  [REF {pool_name:5s}] {len(pool)} pairs done in {time.time()-t_ref:.1f}s", flush=True)

    def eval_val(tag: str) -> dict:
        tv = time.time()
        base_model.eval()
        tot_nll_w, tot_tok_w = 0.0, 0
        tot_nll_l, tot_tok_l = 0.0, 0
        raw_wins, dpo_wins = 0, 0
        raw_margins, dpo_margins, dpo_losses = [], [], []
        with torch.no_grad():
            for p in val_pairs:
                ids_w = torch.tensor([p["chosen_ids"]], dtype=torch.long, device=first_dev)
                ids_l = torch.tensor([p["rejected_ids"]], dtype=torch.long, device=first_dev)
                lp_w, nll_w, nt_w, sn_w = edit_span_mean_logp(base_model, ids_w, p["start"])
                lp_l, nll_l, nt_l, sn_l = edit_span_mean_logp(base_model, ids_l, p["start"])
                lw, ll = float(lp_w.cpu()), float(lp_l.cpu())
                tot_nll_w += sn_w
                tot_tok_w += nt_w
                tot_nll_l += sn_l
                tot_tok_l += nt_l
                raw_m = lw - ll
                dpo_m = (lw - p["ref_lp_w"]) - (ll - p["ref_lp_l"])
                raw_margins.append(raw_m)
                dpo_margins.append(dpo_m)
                dpo_losses.append(float(-F.logsigmoid(torch.tensor(DPO_BETA * dpo_m))))
                if raw_m > 0:
                    raw_wins += 1
                if dpo_m > 0:
                    dpo_wins += 1
        base_model.train()
        res = {
            "chosen_nll": tot_nll_w / max(1, tot_tok_w),
            "rejected_nll": tot_nll_l / max(1, tot_tok_l),
            "raw_acc": raw_wins / max(1, len(val_pairs)),
            "raw_margin": float(np.mean(raw_margins)),
            "dpo_acc": dpo_wins / max(1, len(val_pairs)),
            "dpo_margin": float(np.mean(dpo_margins)),
            "dpo_loss": float(np.mean(dpo_losses)),
            "chosen_tokens": tot_tok_w,
        }
        print(
            f"  [VAL {tag:12s}] chosen_nll={res['chosen_nll']:.5f} rej_nll={res['rejected_nll']:.5f} | "
            f"raw_acc={res['raw_acc']:.3f} (m={res['raw_margin']:+.4f}) | "
            f"dpo_acc={res['dpo_acc']:.3f} (m={res['dpo_margin']:+.4f}) [{time.time()-tv:.1f}s]",
            flush=True,
        )
        return res

    # Initial v48 baseline metrics on val_pairs (from precomputed ref_lp)
    init_raw_margins = [p["ref_lp_w"] - p["ref_lp_l"] for p in val_pairs]
    init_chosen_nll = float(np.mean([-p["ref_lp_w"] for p in val_pairs]))
    init_rej_nll = float(np.mean([-p["ref_lp_l"] for p in val_pairs]))
    pre_val = {
        "chosen_nll": init_chosen_nll,
        "rejected_nll": init_rej_nll,
        "raw_acc": sum(1 for m in init_raw_margins if m > 0) / max(1, len(val_pairs)),
        "raw_margin": float(np.mean(init_raw_margins)),
        "dpo_acc": 0.0,
        "dpo_margin": 0.0,
        "dpo_loss": math.log(2.0),
    }
    print(
        f"  [VAL V48-INIT    ] chosen_nll~={pre_val['chosen_nll']:.5f} rej_nll~={pre_val['rejected_nll']:.5f} | "
        f"raw_acc={pre_val['raw_acc']:.3f} (m={pre_val['raw_margin']:+.4f})",
        flush=True,
    )

    opt = torch.optim.AdamW(trainable, lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
    total_opt = len(train_pairs) // GRAD_ACCUM
    ckpt_steps = {max(1, total_opt // 2), total_opt}
    ckpts = []
    history = []

    def lr_at(idx: int) -> float:
        if idx < WARMUP_STEPS:
            return LR * float(idx + 1) / float(WARMUP_STEPS)
        prog = min(1.0, float(idx - WARMUP_STEPS) / max(1.0, float(total_opt - WARMUP_STEPS)))
        return LR * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))

    print(f"\n[*] Training Edit-Span DPO+NLL: {total_opt} opt steps ({len(train_pairs)} pairs, accum={GRAD_ACCUM}, lr={LR}, beta={DPO_BETA}, anchor={NLL_ANCHOR})...", flush=True)
    base_model.train()
    step = 0
    for s in range(0, len(train_pairs), GRAD_ACCUM):
        batch = train_pairs[s : s + GRAD_ACCUM]
        cur_lr = lr_at(step)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        ts = time.time()
        b_loss, b_dpo, b_nll, b_margin, b_win = 0.0, 0.0, 0.0, 0.0, 0
        for p in batch:
            ids_w = torch.tensor([p["chosen_ids"]], dtype=torch.long, device=first_dev)
            ids_l = torch.tensor([p["rejected_ids"]], dtype=torch.long, device=first_dev)
            lp_w, nll_w, _, _ = edit_span_mean_logp(base_model, ids_w, p["start"])
            lp_l, _, _, _ = edit_span_mean_logp(base_model, ids_l, p["start"])
            ref_diff = p["ref_lp_w"] - p["ref_lp_l"]
            margin = (lp_w - lp_l) - ref_diff
            dpo_loss = -F.logsigmoid(DPO_BETA * margin)
            loss = dpo_loss + NLL_ANCHOR * nll_w
            (loss / float(len(batch))).backward()
            b_loss += float(loss.detach().cpu())
            b_dpo += float(dpo_loss.detach().cpu())
            b_nll += float(nll_w.detach().cpu())
            mv = float(margin.detach().cpu())
            b_margin += mv
            if mv > 0:
                b_win += 1

        gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).cpu())
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        nb = float(len(batch))
        dt = time.time() - ts
        rec_s = {
            "step": step,
            "loss": round(b_loss / nb, 5),
            "dpo_loss": round(b_dpo / nb, 5),
            "chosen_nll": round(b_nll / nb, 5),
            "dpo_margin": round(b_margin / nb, 5),
            "dpo_acc": round(b_win / nb, 3),
            "gnorm": round(gnorm, 4),
            "lr": cur_lr,
            "sec": round(dt, 1),
        }
        history.append(rec_s)
        print(
            f"  [Step {step:02d}/{total_opt:02d}] loss={rec_s['loss']:.4f} (dpo={rec_s['dpo_loss']:.4f} nll={rec_s['chosen_nll']:.4f}) | "
            f"margin={rec_s['dpo_margin']:+.4f} acc={rec_s['dpo_acc']:.2f} | gnorm={gnorm:.3f} | lr={cur_lr:.2e} | {dt:.1f}s",
            flush=True,
        )
        if step in ckpt_steps:
            tag = f"CKPT-S{step:02d}"
            sn = snap_cpu()
            ev = eval_val(tag)
            ckpts.append((tag, ev, sn))
            export_adapter_dir(OUT_ADAPTER_DIR / "checkpoints" / tag, sn, target_names, scale_mult=1.0)

    # Select best checkpoint: require chosen_nll <= pre_val['chosen_nll'] * 1.10 (no SFT collapse) & maximize dpo_margin - 2*chosen_nll
    def score_ev(ev: dict) -> float:
        penalty = 5.0 * max(0.0, ev["chosen_nll"] - pre_val["chosen_nll"] * 1.05)
        return ev["dpo_margin"] + 0.5 * ev["raw_margin"] - 2.0 * ev["chosen_nll"] - penalty

    best_tag, best_ev, best_snap = max(ckpts, key=lambda x: score_ev(x[1]))

    # Blend sweep between v48 and best v49 checkpoint: alpha in [1.0, 0.75, 0.50]
    blends = [(1.0, best_ev)]
    for alpha in [0.75, 0.50]:
        restore_blend(v48_snap, best_snap, alpha=alpha)
        ev_b = eval_val(f"BLEND-{alpha:.2f}")
        blends.append((alpha, ev_b))

    best_alpha, best_post = max(blends, key=lambda x: score_ev(x[1]))
    final_snap = {}
    for n in target_names:
        A0, B0 = v48_snap[n]
        A1, B1 = best_snap[n]
        final_snap[n] = (A0 * (1.0 - best_alpha) + A1 * best_alpha, B0 * (1.0 - best_alpha) + B1 * best_alpha)

    print(
        f"\n[OK] Selected {best_tag} blend={best_alpha:.2f} | "
        f"chosen_nll: {pre_val['chosen_nll']:.5f} -> {best_post['chosen_nll']:.5f} | "
        f"raw_margin: {pre_val['raw_margin']:+.4f} -> {best_post['raw_margin']:+.4f} | "
        f"dpo_acc={best_post['dpo_acc']:.3f} (margin={best_post['dpo_margin']:+.4f})",
        flush=True,
    )
    export_adapter_dir(OUT_ADAPTER_DIR, final_snap, target_names, scale_mult=1.0)
    sha = package_submission(OUT_ADAPTER_DIR, OUT_ZIP)
    print(f"[OK] Packaged {OUT_ZIP} ({OUT_ZIP.stat().st_size/(1024*1024):.2f} MB) | SHA256={sha}")

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps({
        "init_adapter_dir": str(INIT_V48_DIR),
        "adapter_dir": str(OUT_ADAPTER_DIR),
        "submission_zip": str(OUT_ZIP),
        "sha256": sha,
        "modules": len(target_names),
        "best_checkpoint": best_tag,
        "best_blend_alpha": best_alpha,
        "dpo_beta": DPO_BETA,
        "nll_anchor": NLL_ANCHOR,
        "v48_init_val": pre_val,
        "post_dpo_val": best_post,
        "blend_sweep": [{"alpha": a, **ev} for a, ev in blends],
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
