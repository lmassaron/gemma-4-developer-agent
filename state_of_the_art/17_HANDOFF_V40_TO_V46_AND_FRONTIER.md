# Handoff & Current Frontier Briefing (`v40` -> `v46` LoRA)

> **Purpose**: Complete state-of-the-art summary, experimental results, harness quirks, and next frontier directions as of **2026-11-19**, written so another Antigravity instance can resume work seamlessly with zero context loss.

---

## 1. Current Workspace & Submission State

- **Git HEAD**: `a08441d` (`main`, pushed to `origin/main`).
- **Uncommitted `v46` & live workspace artifacts**:
  - `submission_v46_patchsmith_lora.zip` (`192,272,427` B / `183.37 MB`, SHA-256 `547acf1077679e84000a3b9655f4bd9bee037655211b61324419a370bf34c683`) — built and passed `data/dmitrii_kit/check_submission.py` (`result: rules passed`). Ready to copy (`cp submission_v46_patchsmith_lora.zip submission.zip`) if submitting to Kaggle next.
  - `models/lora_v46_patchsmith/` — exported `CKPT-S41` x `1.0` (`r=16, lora_alpha=8, scale=0.50`, `410` `language_model` modules, `820` `bfloat16` tensors), plus saved checkpoints under `models/lora_v46_patchsmith/checkpoints/{CKPT-S16,CKPT-S32,CKPT-S41}/`.
  - `scripts/train_v46_patchsmith_lora.py` — reproducible curation, training, held-out NLL/scale-sweep, and ZIP packaging pipeline.
  - `state_of_the_art/benchmarks_data/v46_lora_training.log`, `v46_lora_training_report.json`, `benchmark_v46_core12.{json,log}`.
  - Updated `LEADERBOARD.md` and `leaderboard.json`.
  - Note: old `submission_v37`–`submission_v40` zips in `submissions/` were removed locally (tracked deletions committed in `a08441d`); root `v42`–`v45` zips can be regenerated anytime via `scripts/package_v4{2,3,4,5}_candidate.py`.
  - Untracked legacy items to keep out of git: `models/lora_v33_native_causal/` (`102M`) and `notebook_artifacts/` (contains `best/score_022_submission.zip` and `controls/exp021_*.zip`).

---

## 2. Kaggle Public Leaderboard History (`v23` – `v45`)

Observed user Kaggle Public LB scores vs. public `exp021` reference (`N ≈ 60–72` public tasks; `0.18` vs `0.22` is ~`2–3` tasks):

| Candidate | Type | Tools | `timeout_seconds` | Prompt / Config Delta | Public LB |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **`v23`** (`bundle_v23_zero_shot`) | Zero-shot | 9 | 240s | `T=0.2`, `thinking_budget=4096`, `8m`, `28` calls | **`0.18`** |
| **`v37`** (`v23` restored + tailguard) | Zero-shot | — | — | Restored `v23` + tailguard | `0.17` |
| **`v40`** (`budget_surgical_v2`) | Zero-shot | 6 | 240s | `Zero pre-edit probes`, strict `Calls 1-7/8-10/11-17/18+` | `0.17` |
| **`v42`** (`patchsmith_surgical`) | Zero-shot | 6 | 60s | Patchsmith V7 + `v40/v41` rules (`5,259 B` prompt) | **`0.18`** |
| **`v45`** (`exp021_surgical`) | Zero-shot | 6 | 60s | `exp021` verbatim (`4,892 B`) + 3 fixes (`5,018 B`) | **`0.18`** |
| **Public `exp021`** (`score_022_submission.zip`) | Zero-shot | 6 | 60s | Original Patchsmith V7 (`4,892 B`), `T=0.2`, `tb=4096` | **`0.22`** |
| **Public `flexonafft`** (Nov 13) | Zero-shot | 6 | 240s | Same `exp021` prompt with `thinking_budget=1024`, `10m` | `0.17` |

**Core conclusion**: Zero-shot prompt tuning around `exp021` / `Budget-Fit` hits a **`0.17–0.22` ceiling** (`0.18` across `v23`, `v42`, `v45`, with ~`±0.04` binomial/sampling variance at `T=0.2`).

---

## 3. Local Benchmark Progression (`v32` – `v46`)

All fast-mode benchmarks run `enable_thinking=False` with `max_new_tokens <= 1024` unless `--respect-thinking` is passed.

### A. Zero-Shot & Post-`v40` Iteration (`leaderboard.json` + custom suites)

| Run | Suite | Mode | Resolved | Avg Dur | Timeouts | Avg Turns | Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`v40`** | `core_12` | fast | **`8/12` (`66.7%`)** | `366.1s` | 3 | 12.2 | Local `core_12` flagship |
| **`v40`** | `heldout_12` (`/tmp/v40_heldout12.jsonl`) | fast | **`11/12` (`91.7%`)** | `141.8s` | 0 | 7.2 | Missed only `fastapi_14372` |
| **`v40`** | `nearmiss_hard12` (`/tmp/v40_nearmiss_hard12.jsonl`) | fast | **`5/12` (`41.7%`)** | `448.4s` | 5 | 12.7 | Grep loops, `curl`, unverified edits |
| **`v40`** | `sympy12` (`data/external_hf_benchmark_py312.jsonl`) | fast | **`5/12` (`41.7%`)** | `502.3s` | 7 | 14.1 | `3` zero-byte diffs; `repro.py` timeouts |
| **`v41`** | `core_12` / `nearmiss_hard12` | fast | `7/12` / `3/12` | `412s` / `383s` | 5 / 4 | 12.8 / 13.3 | Regressed vs `v40` |
| **`v42`** | `core_12` | fast | `4/12` (`33.3%`) | `469.3s` | 8 | 12.5 | `| tail -n 45` & test-hiding regressions |
| **`v42`** | `core_12` | `--respect-thinking` (`480s`) | `1/12` (`8.3%`) | `660.9s` | 11 | 5.8 | Local `bf16` timed out in 4–8 turns |
| **`v43`** | `core_12` | fast | `6/12` (`50.0%`) | `437.0s` | 5 | 12.7 | Pre-edit `python3 -c` probes returned |
| **`v44`** | `core_12` | fast | **`8/12` (`66.7%`)** | `395.7s` | 4 | 12.7 | Tied `v40` on `core_12` |
| **`v44`** | `nearmiss_hard12` | fast | `4/12` (`33.3%`) | `403.4s` | 4 | 13.1 | Regressed `fastapi_14873` (`repro.py`) |
| **`v44`** | `sympy12` | `--respect-thinking` (`480s`) | `2/12` (`16.7%`) | `723.4s` | 10 | 6.3 | Wrote `repro*.py` on 5/12 tasks |
| **`v45`** | `smoke_3` | `--respect-thinking` (`1800s`) | `1/3` (`33.3%`) | `2309.5s` | 2 | 12.0 | `requests_7205` solved; `/tmp/repro.py` stalls |

### B. LoRA Progression (`v28` – `v46`)

| Run | Data / Config | Scale (`alpha/r`) | `core_12` Resolved | Notes |
| :--- | :--- | :---: | :---: | :--- |
| `v28`–`v30` | External / community LoRAs | varied | `1/12`–`3/12` (`8.3%–25.0%`) | Format/reasoning mismatch |
| `v33` | 32 native steps (8 opt steps), `r=8, alpha=4`, `0.0015` norm choke | `0.50` (choked) | `4/12` (`33.3%`) | Pre/post NLL flat (`0.12393 -> 0.12402`) |
| `v35` | `lora_v33` rescaled (`alpha=2`) + lean6 prompt | `0.25` | **`6/12` (`50.0%`)** | Best local `core_12` LoRA (`438.6s`) |
| **`v46`** | `328` curated Patchsmith-aligned native steps (`41` opt steps, `r=16, alpha=8`) | `0.50` | **`4/12` (`33.3%`)** | `529.2s`; Gate A NLL dropped `-4.5% / -13.4%` |

---

## 4. Critical Engineering Findings (`v42` – `v46`)

1. **Local vs. Kaggle Inversion (`enable_thinking=False` vs. `True`)**:
   - Under local fast mode (`enable_thinking=False`), `v40`/`v44` (`Zero pre-edit probes`) score `8/12` on `core_12`, while `Patchsmith` (`v42`, `v46`) scores `4/12`. On Kaggle (`thinking_budget=4096`), `v42`/`v45` scored `0.18` vs `v40`'s `0.17` (`exp021`: `0.22`).
   - `scripts/run_local_benchmark_leaderboard.py` now supports `--respect-thinking` (enables `enable_thinking=True`, raises `max_new_tokens` cap to `8192`, closes `<channel|>` at `thinking_budget` via `ThinkingBudgetLogitsProcessor`, preserves `top_k`) and `--timeout_override` (overrides `eval_config.yaml` `max_time_minutes`).
   - On local `NVIDIA GB10` (`bfloat16` 31B), `--respect-thinking` runs at ~`80–190s/turn`, so Kaggle's `480s` timeout stops runs after ~`5–9` turns (`1/12` for `v42`, `2/12` for `v44` `sympy12`). Use `--timeout_override 1800` for `smoke_3` thinking diagnostics (~12 turns, `0` unparsed turns).

2. **Four Regressions Found in `v42` vs `exp021` (Fixed in `v45` / `v46`)**:
   - `v42` added `| tail -n 45` to pytest commands (masked non-zero exit codes as `0`!).
   - `v42` added `':!tests/' ':!docs/' ':!docs_src/' | head -n 30` to `git grep` (hid test clues & truncated matches).
   - `v42` added rigid `<= 15-20 changed lines` and `60-100` line caps (`5,259 B` vs `exp021`'s `4,892 B`).
   - `v42` omitted `exp022`'s `If edit_file returns status: ok, do not repeat it, regardless of strategy` clarity.

3. **Python Environments & `Gemma4ClippableLinear` Trap**:
   - `.venv/bin/python` (`torch 2.14.1`, `transformers 5.20.2`, `peft 0.21.2`) runs local evaluation on `google/gemma-4-31B-it`.
   - `/home/lmassaron/code/venv_gemma4_12b/bin/python` has `compressed-tensors==0.20.1` required to load `/home/lmassaron/models/kaggle/gemma-4-31b-it-qat-w4a16-ct` in packed INT4 via `Int4LoRALinear`.
   - `Gemma4ForConditionalGeneration` uses `Gemma4ClippableLinear` inside `vision_tower` (`1152`-dim). Never pass bare projection suffixes like `["q_proj", ...]` without filtering to `model.language_model.layers.\d+\....`; `v46` lists all `410` `language_model` projection paths explicitly (`820` tensors; `10` layers share KV without `v_proj`), loading cleanly in `PeftModel.from_pretrained` with zero errors.

4. **Dataset Findings in `data/dmitrii_kit/`**:
   - **`lora_native_resolved.jsonl`** (`103` trajectories, `1,766` steps across `52` repos) has **zero repository overlap** with `psf/requests`, `fastapi/fastapi`, `Textualize/rich`, or `sympy/sympy`. All `1,766` steps use only the 6 core tools (`0` graph tool calls).
   - Auditing turn $t$ against turn $t+1$'s tool output uncovered substantial hidden noise inside resolved native episodes: `173` `read_file` errors, `123` duplicate/forbidden commands, `113` `write_file` scratch/repro/test scripts, `89` empty-fail commands, `14` `edit_file` errors, and `145` post-submit prose turns.
   - **`lora_external_A.jsonl` (`382` steps) & `lora_external_B.part*.jsonl` (`25,165` steps)** have empty thought channels (`<|channel>thought\n<channel|>`). Supervising empty thoughts would collapse Kaggle `thinking_budget=4096` reasoning (`flexonafft` drop `0.22 -> 0.17`).
   - **`lora_native_unresolved.jsonl`** (`56` failed trajectories, `1,228` steps) is available for contrastive/DPO preference pairs against failure loops.

---

## 5. Deep Dive on `v46` (`submission_v46_patchsmith_lora.zip`)

- **Pipeline ([train_v46_patchsmith_lora.py](file:///home/lmassaron/code/gemma-4-developer-agent-competition/scripts/train_v46_patchsmith_lora.py))**:
  - Replaced `2,328`-token native 9-tool system prefix with exact `1,461`-token `V45_SYSTEM_MD` 6-tool Patchsmith prefix while preserving served `completion_token_ids` verbatim.
  - Filtered `<= 10,240` tokens (zero truncation): **`328` train steps** (`99` command, `83` read, `77` edit, `69` submit across `43` repos -> `41` opt steps at `grad_accum=8`, `lr=1.5e-5`) and **`58` held-out validation steps** across `8` repos.
  - `ACTION_PHASE_WEIGHT=1.25`, `THOUGHT_PHASE_WEIGHT=0.90`, `r=16, lora_alpha=8` (`scale=0.50`).
- **Gate A (`enable_thinking=True` Held-Out Repo NLL)**:
  - `PRE-TRAIN`: `Overall=0.11445 | Action=0.01751 | Thought=0.14852`
  - `CKPT-S16`: `Overall=0.11333 | Action=0.01676 | Thought=0.14727`
  - `CKPT-S32`: `Overall=0.10992 | Action=0.01540 | Thought=0.14313`
  - `CKPT-S41` (`1.0x`, `scale=0.50`): **`Overall=0.10931 (-4.5%) | Action=0.01516 (-13.4%) | Thought=0.14239 (-4.1%)`** (strictly beat `0.75x`, `0.50x`, `0.25x` in scale sweep).
- **Gate B (`core_12`, `enable_thinking=False`, [benchmark_v46_core12.json](file:///home/lmassaron/code/gemma-4-developer-agent-competition/state_of_the_art/benchmarks_data/benchmark_v46_core12.json))**:
  - **`4/12` (`33.33%`)**, `529.21s` avg, `8` timeouts, `14.5` avg turns (vs `v42` `4/12`, `469.33s`, `8` timeouts, `12.5` turns).
  - **Positives**: `6/12` non-empty source patches (vs `4/12` in `v42`); gained `fastapi_13537` (`0B -> 828B RESOLVED_ANYTIME`), sped up `rich_3006` (`496.5s RESOLVED_ANYTIME -> 394.5s RESOLVED`), applied multi-site edits on `requests_7505` (`507B`) and `rich_3052` (`1450B`).
  - **Remaining flaws**: regressed `requests_7433` (`223s RESOLVED -> 548.5s TIMEOUT 0B` stuck editing `/tmp/repro.py`), `fastapi_12942` hit a grep loop (`24` `[CIRCUIT BREAKER]` triggers total in the run), and `529.2s` avg duration locally.

---

## 6. Current Frontier & Concrete Next Steps

1. **Immediate Action (Pending `v46` Kaggle LB Result)**:
   - Submit `submission_v46_patchsmith_lora.zip` (copied as `submission.zip`) or run a quick `smoke_3 --respect-thinking --timeout_override 1800` check vs `v45` (`1/3`) to see how `v46` behaves under its native `enable_thinking=True` distribution.

2. **Frontier A — Contrastive / DPO LoRA (`v47`) Against Actual Observed Failure Modes**:
   - Positive SFT alone (even after filtering `lora_native_resolved.jsonl`) still allowed `v46` to loop on `/tmp/repro.py` (`requests_7433`), repeat `grep` (`fastapi_12942`), and spend early turns on scratch probes.
   - Use **`data/dmitrii_kit/lora_native_unresolved.jsonl`** (`56` failed episodes, `1,228` steps) + local `benchmark_v4*` failure logs (`rich_4075`, `fastapi_11355`, `fastapi_12942`, `requests_7433`) to build DPO / SimPO / unlikelihood preference pairs:
     - **Chosen**: bounded targeted reads / grounded `edit_file` / single verification -> `submit_patch`.
     - **Rejected**: duplicate identical commands, `/tmp/repro.py` iteration loops, `write_file` scratch scripts, ungrounded edits, `0B` `submit_patch`.
   - Start from `models/lora_v46_patchsmith/checkpoints/CKPT-S41` (or `CKPT-S32`) and gate on **both** `smoke_3 --respect-thinking --timeout_override 1800` and `core_12` (`> 4/12`, fewer `[CIRCUIT BREAKER]` triggers, lower avg duration).

3. **Frontier B — Action-Only Masked SFT on `external_B` (`729` Trajectories Across Unseen Repos)**:
   - `lora_external_B.part000..003.jsonl` provides `13,110` steps (`<= 16k` tokens) across `729` resolved issues — **13x larger** repo diversity than `native_resolved`.
   - Crucially: **mask the empty `<|channel>thought\n<channel|>` header (`-100`)** on `external_B` and supervise only `<|tool_call>...<tool_call|>` tokens, mixed `1:1` with full thought+action steps from `lora_native_resolved.jsonl` (fixing prompt alignment + filtering next-turn errors just like `v46`).

4. **Frontier C — Kaggle Budget & Sampling Calibration on `exp021` / `v46`**:
   - Sweep `scale_mult` (`0.25` vs `0.50` — note `v35` hit `6/12` on `core_12` at `scale=0.25` vs `v33`'s `4/12`) on `smoke_3` thinking.
   - Test `thinking_budget: 2048` or `3072` (reducing per-turn latency so 28 tool calls complete inside `8m` / `12h` without `flexonafft`'s `1024`-token collapse) and `temperature: 0.0` / `0.1` + `top_k: 64` (reducing stochastic misses at `T=0.2`).
   - Also keep `notebook_artifacts/best/score_022_submission.zip` (`exp021` verbatim) as an optional zero-shot control baseline.

---

## 7. Quick Command Reference

```bash
# 1. Regenerate v45 zero-shot zip or validate any candidate zip
.venv/bin/python scripts/package_v45_candidate.py
cp submission_v46_patchsmith_lora.zip /tmp/submission.zip && \
  .venv/bin/python data/dmitrii_kit/check_submission.py /tmp/submission.zip

# 2. Train LoRA on packed QAT 31B weights (MUST use venv_gemma4_12b for compressed-tensors==0.20.1)
/home/lmassaron/code/venv_gemma4_12b/bin/python scripts/train_v46_patchsmith_lora.py --dry-run
PYTHONUNBUFFERED=1 /home/lmassaron/code/venv_gemma4_12b/bin/python -u scripts/train_v46_patchsmith_lora.py

# 3. Local evaluation: fast mode vs faithful thinking mode (uses .venv)
PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/run_local_benchmark_leaderboard.py \
  --suite core_12 --submission_zip submission_v46_patchsmith_lora.zip \
  --model_name v46_patchsmith_lora --output_file out.json

PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/run_local_benchmark_leaderboard.py \
  --suite smoke_3 --respect-thinking --timeout_override 1800 \
  --submission_zip submission_v46_patchsmith_lora.zip \
  --model_name v46_smoke3_thinking --output_file out_think.json
```
