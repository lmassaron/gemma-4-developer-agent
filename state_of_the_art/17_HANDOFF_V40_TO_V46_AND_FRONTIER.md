# Handoff & Current Frontier Briefing (`v40` -> `v49` Edit-Span DPO)

> **Purpose**: Complete state-of-the-art summary, verified Kaggle LB & local benchmark tables, failure EDA, environment quirks, and concrete final-week playbook as of **2026-11-21**, written so another Antigravity instance can resume seamlessly with zero context loss.

---

## 1. Competition Deadlines, Live LB & Current Workspace State

- **Competition**: `gemma-4-developer-agent` (*Google – The Gemma 4 Developer Agent Competition*)
- **Deadlines**: Team Merger **2026-11-25** | Final Submission **2026-12-02** (**11 days remaining**).
- **Exact Public LB Structure (verified Nov 20 via `kaggle==2.2.5` API & `dariushafshar` analysis)**:
  - Public LB has **exact `N = 58` tasks**, truncated to 2 decimals (**`1` task = `0.01724`**): `9/58=0.15`, `10/58=0.17`, `11/58=0.18`, `12/58=0.20`, `13/58=0.22`, `14/58=0.24`.
  - Live Nov 20 top leaderboard: **Top 16 teams (`#1–#16`) are all tied at `0.24` (`14/58`)**, `#17–#20+` at `0.22` (`13/58`). Eleven of the 16 teams at `0.24` submitted Nov 18–20 right after the Nov 14–17 `40`-call / pinned-issue notebooks (`bliverrigpdoge`, `denpugovkin exp025`, `Patchsmith V11`).
  - Single-draw stochastic SD at `T = 0.2` is **~`1.43–1.78` tasks (`±0.025–0.031`)** — proven directly by our own control: byte-for-byte `exp021` (`score_022_submission`, `ref 58582549`) scored **`0.17` (`10/58`)** on our account vs `0.22` (`13/58`) on `denpugovkin`'s draw.
- **Ready, Validated Local Submission Zips (all passed `data/dmitrii_kit/check_submission.py` -> `result: rules passed`)**:
  1. **`submission_v47_patchsmith_v11_pinned40.zip`** (`6,046 B`, SHA-256 `e330227f2a8676a19338f96b3d9629cb684f049e2cc905fdfb6b13adc7317c3f`) — **SUBMITTED TO KAGGLE (`ref 58637088`, `2026-11-20 21:50 UTC`, currently `PENDING`)**.
  2. **`submission_v49_patchsmith_v11_edit_dpo.zip`** (`40.55 MB`, SHA-256 `98a445174baf8366cbd6038518a4f0defb095cf0c5990819f989c249c8d6a1a1`) — **RECOMMENDED NEXT KAGGLE UPLOAD** (`v47` + upper-30-layer 90-module Edit-Span DPO+NLL LoRA initialized from `v48`; `5/12` clean `RESOLVED`, `10.67` avg turns).
  3. **`submission_v48_patchsmith_v11_edit_lora.zip`** (`40.54 MB`, SHA-256 `b46b99a1083a0b39bbbe23c551a99925cc75853213aba634de2b76eb08d11d57`) — **RECOMMENDED FOLLOWING KAGGLE UPLOAD** (`v47` + upper-30-layer 90-module Edit-Only SFT LoRA; `6/12` total `core_12`, `4` clean + `2` anytime).
- **Untracked large artifacts (intentionally kept out of git)**:
  - `models/lora_v33_native_causal/`, `models/lora_v46_patchsmith/`, `models/lora_v48_edit_only/`, `models/lora_v49_edit_dpo/` (plus `checkpoints/` inside each), `notebook_artifacts/`, and root `submission_*.zip` files.

---

## 2. Kaggle Public Leaderboard History (`v23` – `v47`)

| Ref / Candidate | Type | Tools | Call / Time Budget | Key Delta | Public LB |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **`v23`** (`bundle_v23_zero_shot`) | Zero-shot | 9 | `28` calls / `8m` / `240s` | `T=0.2`, `tb=4096` | **`0.18`** (`11/58`) |
| **`v34`** (`clean024_mandatory_verify`) | Zero-shot | 6 | `28` calls / `5m` | Cutting time cap to `5m` lost `49%` of solvable runs | `0.10` (`6/58`) |
| **`v37`** (`v23` restored + tailguard) | Zero-shot | 9 | `28` calls / `8m` | Restored `v23` + tailguard | `0.17` (`10/58`) |
| **`v40`** (`budget_surgical_v2`) | Zero-shot | 6 | `28` calls / `8m` / `240s` | Zero pre-edit probes, strict call phases | `0.17` (`10/58`) |
| **`v42`** (`patchsmith_surgical`) | Zero-shot | 6 | `28` calls / `8m` / `60s` | Patchsmith V7 + `v40/v41` rules (`5,259 B`) | **`0.18`** (`11/58`) |
| **`v45`** (`exp021_surgical`) | Zero-shot | 6 | `28` calls / `8m` / `60s` | `exp021` verbatim + 3 fixes (`5,018 B`) | **`0.18`** (`11/58`) |
| **`v46`** (`patchsmith_lora`) | Full-turn LoRA | 6 | `28` calls / `8m` / `60s` | `410` modules (all 60 layers), full thought+action SFT (`183 MB`) | `0.15` (`9/58`) |
| **`score_022_submission`** (`58582549`) | Zero-shot control | 6 | `28` calls / `8m` / `60s` | Byte-for-byte `exp021` (`4,892 B`) — variance control | `0.17` (`10/58`) |
| **`v47`** (`58637088`, Nov 20 21:50 UTC) | Zero-shot | 6 | **`40` calls / `8m` / `60s`** | Patchsmith V11 + compaction-pinned `{problem_description}`/`{hints?}` + fast exit | **`PENDING`** |

---

## 3. Local Benchmark Progression (`v40` – `v49`)

All default `core_12` runs execute `enable_thinking=False` (`max_new_tokens <= 1024`) on `google/gemma-4-31B-it` (`bfloat16` on local `NVIDIA GB10`).

### A. Flagship 5-Way Paired Progression on `core_12` (`v42` -> `v49`)

| Run | Total Resolved | Clean `RESOLVED` (`submit_patch`) | `ANYTIME` | Mean Dur | Mean Turns | Timeouts | `edit_file` Calls | Unparsed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`v42`** zero-shot (`0.18` LB) | `4 / 12` (`33.3%`) | `2` | `2` | `469.3s` | `12.50` | `8` | — | `0` |
| **`v46`** full-turn 410-mod LoRA (`0.15` LB) | `4 / 12` (`33.3%`) | `2` | `2` | `529.2s` | `14.50` | `8` | — | `0` |
| **`v47`** zero-shot (`40` calls + pinned) | `5 / 12` (`41.7%`) | `1` | `4` | `509.6s` | `13.50` | `7` | `12` | `0` |
| **`v48`** edit-only SFT LoRA (`90` mod) | **`6 / 12` (`50.0%`)** | `4` | `2` | `530.6s` | `11.75` | `6` | **`19`** | `0` |
| **`v49`** edit-only DPO+NLL LoRA (`90` mod) | `5 / 12` (`41.7%`) | **`5`** | **`0`** | **`490.2s`** | **`10.67`** | `7` | `17` | **`0`** |

**Per-Task Diagnostic (`v47` vs `v48` vs `v49`)**:
- **Shared 5 solves (`requests_7205`, `requests_7433`, `requests_7505`, `rich_2943`, `rich_3006`)**: `v49` cleanly called `submit_patch` on **all 5 (`5/5`, `0` `ANYTIME`)**, averaging **`342s` and `11.2` turns** vs `v48`'s `406s` / `12.6` turns (`4/5` clean) and `v47`'s `1/5` clean (`requests_7433` timed out). Notably, `v49` is the **only model ever** to cleanly resolve `requests_7205` before timeout (`381s` vs `v48` `554s` `ANYTIME`), and cut `requests_7433` to `234s` (`8` turns).
- **Sole `v48` vs `v49` discrepancy (`fastapi_13537`)**: Log diff confirmed `v48` and `v49` followed the **exact same 7-turn trajectory** (`grep _extract_form_body` -> read `utils.py` -> write/run/edit `/tmp/repro.py`); `v49` simply hit local GB10 `bf16` `480s` timeout after Turn 7 (`546s`), while `v48` started Turn 9 at `475s` (`703s` `ANYTIME`). On Kaggle vLLM GPUs, both reach Turn 9 in under a minute.
- **Historical non-`core_12` references**: `v40` (`core_12` fast `8/12`, `heldout_12` `11/12`, `nearmiss_hard12` `5/12`, `sympy12` `5/12`); `v44` (`core_12` `8/12`, `nearmiss_hard12` `4/12`, `sympy12` thinking `2/12`); `v42` (`core_12` thinking `1/12`); `v45` (`smoke_3` thinking `1800s` `1/3`, `0` unparsed).

---

## 4. Key Discoveries & Why `v47` / `v48` / `v49` Work

### A. Why `v47` Improves Structurally Over `exp021` (`28` calls)
Built by [package_v47_candidate.py](file:///home/lmassaron/code/gemma-4-developer-agent-competition/scripts/package_v47_candidate.py):
1. **`max_tool_calls: 40`** (`max_time_minutes: 8`, `timeout_seconds: 60`, `max_turns: 80`, `T=0.2`, `top_p=0.95`, `max_output_tokens=8192`, `thinking_budget=4096`): removes the `28`-call cutoff on multi-file fixes while keeping `8m` (`95.6%` of solvable runs vs `51.1%` at `5m`).
2. **Compaction-proof `{problem_description}` + `{hints?}` pinned at bottom of `prompts/system.md`**: ADK compacts conversation history around ~`14,000` tokens; valid ADK state keys `{problem_description}` and `{hints?}` (the ONLY allowed `{}` braces in `system.md`) keep exact issue specs intact through calls `15–40`.
3. **Fast exit rules**: immediate `submit_patch` on call-budget exhaustion + single-word `Done` afterward protects valid diffs and 12h wall-clock headroom.

### B. Why `v46` Failed (`0.15` LB) and Why `v48` / `v49` Fix It
- **`v46` (`0.15` LB)** supervised **full turns (thoughts + commands + edits)** across **all 60 layers (`410` modules, `183.4 MB`)** — causing subtle thought/loop drift (`14.5` avg turns, `529s`) and 4.5x higher vLLM LoRA overhead. Japan Open Data Lab's Discussion `760014` (`1,800+` E4B trials) independently echoed that standard full-turn teacher SFT loses to the base model and that disabling thoughts or choking output tokens spikes tool-call JSON/truncation errors.
- **`v48` ([train_v48_edit_only_lora.py](file:///home/lmassaron/code/gemma-4-developer-agent-competition/scripts/train_v48_edit_only_lora.py))** — inspired by `ayush04barai/version-1-added-adaptor` (`0.22` live Nov 20 LB):
  - **Loss strictly on `<|tool_call>call:edit_file{...}` spans on source files** (`loss = 0` on ALL thoughts and bash commands -> zero thought collapse, safely mixes `143` native Gemma + `97` Qwen3-Coder `external_A`/`B` edits across `103` repos = `46,280` edit tokens, `0` eval repo overlap).
  - **Upper 30 layers only (`layers 30..59` × `q_proj`/`o_proj`/`down_proj` = `90` modules, `180` tensors, `40.5 MB` zip — 4.5x smaller/faster than `v46`)**, `r=16, lora_alpha=16`, `max_output_tokens: 6144`.
  - **Gate A ([v48_edit_lora_training_report.json](file:///home/lmassaron/code/gemma-4-developer-agent-competition/state_of_the_art/benchmarks_data/v48_edit_lora_training_report.json))**: Held-out `Edit-Span NLL` (`32` edits across `19` unseen repos, `6,027` tokens) dropped **`-33.0%`** (`0.09266 -> 0.06204` at `CKPT-S40 x 1.00`, strictly monotonic scale sweep `1.00/0.75/0.50/0.25`).
- **Native Gemma 4 `edit_file` Failure EDA (`data/dmitrii_kit`)**:
  - Real native Gemma 4 `edit_file` error rate is **`28.5%` (`103 / 362` calls with tool responses)**: `20.0%` (`26/130`) even on resolved tasks, jumping to **`51.3%` (`59/115`)** on unresolved tasks!
  - Exact error breakdown: **`55.3%` (`57/103`) schema/delimiter corruption** (e.g. Gemma emits backtick `` ` `` instead of `<|"|>` before `,old_string:`, triggering ADK `mandatory input parameters are not present: old_string`), **`39.8%` (`41/103`) `old_string_not_found`** (indentation shift, double-escaped `\\n`, `# ...` ellipsis), **`3.9%` (`4/103`) ambiguous short `old_string`**, **`1.0%` (`1/103`) empty `old_string`**.
  - `lora_external_B2.part000..003.jsonl` also provides **`113` additional clean source edits (`<=12k`)** unseen by `v48`.
- **`v49` ([train_v49_edit_dpo.py](file:///home/lmassaron/code/gemma-4-developer-agent-competition/scripts/train_v49_edit_dpo.py))** — **Edit-Span DPO + SFT NLL Anchor (RPO)** initialized from `v48` (`CKPT-S40`):
  - Precomputes fixed `v48` reference log-probs once (`no_grad()`), trains `16` opt steps (`lr=2e-5`, `β=0.2` on token-mean log-probs, `nll_anchor=0.25`) on **`64` pairs across `47` repos** (`9` real native Gemma 4 self-correction pairs + `28` native + `16` fresh `ext_B2` + `7` `ext_B` + `4` `ext_A`; negatives strictly matched to the 4 real error modes above).
  - **Gate A ([v49_edit_dpo_training_report.json](file:///home/lmassaron/code/gemma-4-developer-agent-competition/state_of_the_art/benchmarks_data/v49_edit_dpo_training_report.json))**: Across `24` held-out pairs (`21` unseen repos, `4,650` chosen edit tokens), `CKPT-S16` (`blend=1.00`) cut `chosen_nll` another **`-16.4%`** (`0.01529 -> 0.01278`), pushed `rejected_nll` **up** (`0.11137 -> 0.11731`), widened clean-vs-broken `raw_margin` by **`+57.3%`** (`+0.0961 -> +0.1511`), and hit **`1.000` (`24/24`) DPO accuracy** (`dpo_margin = +0.0550`).

---

## 5. Do-Not-Repeat List & Environment Traps

1. **Abandoned approaches (do NOT repeat)**:
   - Full-turn SFT LoRA supervising thoughts/commands across all 60 layers (`v4`, `v33`, `v46` -> `0.15` LB).
   - Multi-agent / reviewer / analyzer pipelines (`0.06–0.12` LB; `p=0.0005` worse than single coder).
   - Cutting `max_time_minutes` to `5m` (`0.10` LB) or `thinking_budget` to `1024` (`0.17` LB) or disabling thoughts.
   - Piping pytest through `| tail -n 45` (masks exit codes as `0`) or censoring `':!tests/'` in `git grep`.
2. **Python Environments**:
   - **Evaluation / Kaggle API**: `.venv/bin/python` (`torch 2.14.1`, `transformers 5.20.2`, `peft 0.21.2`, **`kaggle==2.2.5`** using `~/.kaggle/access_token`).
   - **Packed INT4 QAT LoRA training**: MUST use `/home/lmassaron/code/venv_gemma4_12b/bin/python` (`compressed-tensors==0.20.1`, `peft==0.21.2`, `transformers==5.21.0.dev0`) to load `/home/lmassaron/models/kaggle/gemma-4-31b-it-qat-w4a16-ct`.
3. **PEFT `Gemma4ClippableLinear` Trap**: Never use bare `target_modules=["q_proj", ...]`; always write explicit `model.language_model.layers.{i}...` paths into `adapter_config.json` to avoid `vision_tower`.
4. **Remote Host Auto-Shutdown Caveat (`spark-d8ec`)**: `antigravity-ide-server` runs with `--enable-remote-auto-shutdown`. Long overnight runs launched as IDE child processes will be killed if the user closes their MacBook lid and macOS sleeps; use `nohup setsid ... </dev/null &` or `tmux` if the user wants to close the lid overnight.
5. **Gemma 4 Tool-Call Closing Tag**: Exact Gemma 4 tool-call syntax is `<|tool_call>call:edit_file{filepath:<|"|>...<|"|>,new_string:<|"|>...<|"|>,old_string:<|"|>...<|"|>}<tool_call|><|tool_response>` (note `<tool_call|>`, NOT `<|tool_call|>`).

---

## 6. Final-Week Action Plan (Nov 21 -> Dec 2)

1. **Check `v47` Kaggle Score (`ref 58637088`)**:
   - Run `.venv/bin/kaggle competitions submissions -c gemma-4-developer-agent -v | head -n 8`.
   - Confirm `v47` finished cleanly inside the 12-hour global cap (`40` calls + pinned issue) and check its score (`expected 0.20–0.22`, upside `0.24`).
2. **Upload `v49` Next, Then `v48` on the Following Slot**:
   - **Slot 1**: Upload `submission_v49_patchsmith_v11_edit_dpo.zip` (highest clean solve rate `5/12`, fewest turns `10.67`, fastest `490.2s`, `+57.3%` edit preference margin).
   - **Slot 2**: Upload `submission_v48_patchsmith_v11_edit_lora.zip` (`6/12` local `core_12`, pure Edit-SFT control vs `v49` Edit-DPO).
3. **Conditional `v50` Iteration (Based on `v47` / `v48` / `v49` Live LB Scores)**:
   - **If `v47`/`v48`/`v49` shows any 12h timeout risk**: test `thinking_budget: 2048` or `3072` (keeping `max_tool_calls: 40` and `max_output_tokens: 6144`) to recover runtime margin without `1024`-token collapse.
   - **If `v48`/`v49` outperforms `v47` on Kaggle (`>= 0.22`)**:
     - Test **`v49` scaled to `0.75`** (`blend=0.75` or `scale=0.75`, since `ayush04barai` and `icfoer` saw lower scales stabilize long-horizon sampling), OR
     - Run **`v50`: Single-step Verifiable `edit_file` GRPO / Extended Edit-Span DPO** adding `<=16k` `external_B` + `B2` pairs (`MAX_CONTEXT=14336`) and real `v47`/`v48`/`v49` local `core_12` failure pairs (strictly excluding eval repos if benchmarking on `core_12`, or training on `sympy`/held-out repos).
   - **If pure zero-shot `v47` beats LoRA on Kaggle**: tune `v47` sampling (`temperature: 0.1–0.2`, `top_k: 64`, `thinking_budget: 2048–4096`) and take 2–3 draws to separate mean from `±1.5`-task `T=0.2` variance.
4. **Two Diverse Final Selections Before Dec 2**:
   - Select **1 best pure zero-shot (`v47` family)** + **1 best surgical edit-only LoRA (`v48`/`v49`/`v50` family)** as the two final Private LB submissions to hedge against unseen-repo distribution shift.

---

## 7. Quick Command Reference

```bash
# 1. Check live Kaggle submissions & top leaderboard
.venv/bin/kaggle competitions submissions -c gemma-4-developer-agent -v | head -n 10
.venv/bin/kaggle competitions leaderboard -c gemma-4-developer-agent --show | head -n 25

# 2. Validate any submission zip (MUST be copied to /tmp/submission.zip)
cp submission_v49_patchsmith_v11_edit_dpo.zip /tmp/submission.zip && \
  .venv/bin/python data/dmitrii_kit/check_submission.py /tmp/submission.zip

# 3. Retrain v48 (Edit-SFT) or v49 (Edit-DPO) on packed QAT 31B (MUST use venv_gemma4_12b)
/home/lmassaron/code/venv_gemma4_12b/bin/python scripts/train_v48_edit_only_lora.py --dry-run
/home/lmassaron/code/venv_gemma4_12b/bin/python scripts/train_v49_edit_dpo.py --dry-run

# 4. Run local core_12 benchmark (uses .venv; supports --respect-thinking --timeout_override 1800)
PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/run_local_benchmark_leaderboard.py \
  --suite core_12 --submission_zip submission_v49_patchsmith_v11_edit_dpo.zip \
  --model_name "v49-patchsmith-v11-edit-dpo" \
  --output_file state_of_the_art/benchmarks_data/benchmark_v49_core12.json
```
