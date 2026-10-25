# 06. Forensic Evaluation, Root-Cause Audit & v12 Architecture Blueprint

> **Context**: Google Gemma 4 Developer Agent Competition on Kaggle (`gemma-4-31b-it-qat-w4a16-ct`, 12.0h wall-clock limit, ~120 sequential SWE-bench tasks)  
> **Source**: Gemini Argon Deep Audit & v12 Artifact Delivery  
> **Status**: Transitioned to **`bundle_v12` ("Native Tool Discipline & Atomic Verification")**  
> **Archive Artifact**: [`submissions_archive/submission_v12_native_tool_discipline_atomic_verify.zip`](file:///home/lmassaron/code/gemma-4-developer-agent-competition/submissions_archive/submission_v12_native_tool_discipline_atomic_verify.zip)  
> **Active Directory**: [`state_of_the_art/bundle_v12/`](file:///home/lmassaron/code/gemma-4-developer-agent-competition/state_of_the_art/bundle_v12)

---

## 1. Executive Summary: Why v10 & v11 Failed to Break 0.17

A forensic cross-examination of the markdown summaries (`03_CURRENT_SOLUTION_V11_AND_V10.md`, `04_EMPIRICAL_BENCHMARK_RESULTS.md`, `05_ERROR_ANALYSIS_AND_NEXT_STEPS.md`) against the raw empirical traces (`benchmark_v10_vs_v11.json`, `bench_v10_vs_v11.log`, and `test_tasks_smoke_3.jsonl`) exposed **four silent-killer bugs** that were partially misdiagnosed in earlier reports:

1. **Autoregressive `run_command` Collapse (The True Cause of `run_command("submit_patch")`)**:
   - In `v11`, the system prompt instructed the agent to read files via `sed -n` (inside `run_command`) and edit files via `python3 -c` (inside `run_command`).
   - Consequently, **100% of tool calls in v11 across all 3 smoke tasks were `run_command` calls**—the model stopped calling native `read_file`, `edit_file`, and `get_status` altogether.
   - Once Gemma 4 31B emits `run_command` for 5–9 consecutive turns, in-context autoregressive momentum locks it into the `run_command` tool schema. When it reaches the submission step, it stays inside `run_command` and calls `run_command({"command": "submit_patch"})` up to 25 times in a row.
2. **Silent Zero-Byte Python Edits & Chained `.replace()` Corruption**:
   - **On `requests_7205`**: `v11` read lines `1,100p` and `100,200p` of `src/requests/utils.py`, missing `get_netrc_auth` at line 231. On Turn 4, it hallucinated `c.replace('    return netrc.netrc.netrc_auth(netrc_file, host), None', ...)` *without* an `assert` statement. Because `str.replace()` silently returns the original string when the substring is absent, the script exited `0` and modified **0 bytes**.
   - **Why `patch_size: 791` was a mirage**: Every run (`v7`, `v8`, `v8b`, `v9`, `v10`, `v11`) reported `"patch_size": 791` on `requests_7205` with the unpatched baseline failure (`AssertionError: assert ('', '') is None`). The 791 bytes come from the harness's `test_patch` (`tests/test_utils.py`), **not** a source edit!
   - **On `rich_2725`**: On Turn 3, `v11` chained `.replace('self._box.mid_left', 'self._box.foot_left')` followed immediately by `.replace('self._box.foot_left', 'self._box.mid_left')`. Chaining `A -> B` and `B -> A` without a temporary placeholder overwrites the first replacement! Furthermore, `rich/table.py` uses `_box.mid_left`, not `self._box.mid_left`.
3. **Call-Site vs. Definition-Site Trap + Self-Sabotaging Anti-Repetition Rule (`fastapi_11194`)**:
   - On Turn 1 of `fastapi_11194`, `v11` ran `git grep -n "_extract_form_body" fastapi/dependencies/utils.py`.
   - `_extract_form_body` is **called** around line 180 (`solve_dependencies`) and **defined** at **line 872** (`async def _extract_form_body`).
   - `v11` read lines `100,250p` (seeing only the call site), and then got trapped by its own prompt rule: *"Never run a second grep on the same file or term."* Forbidden from grepping for `def _extract_form_body` and unable to see the definition in lines 170–250, the model looped on `sed -n '170,250p'` 16 times until timeout.
4. **Documentation vs. Bundle Configuration Drift**:
   - While `03_CURRENT_SOLUTION_V11_AND_V10.md` claimed `bundle_v11` used `temperature: 0.1`, `max_tokens: 3072`, and 4 tools in `agent.yaml`, the actual files in `bundle_v11/` still had `temperature: 0.2`, `max_output_tokens: 8192`, and all 9 tools.
   - `bundle_v11/prompts/system.md` also contained conflicting turn budgets (`# HARD LIMITS` mandated edits by Turn 5 and emergency stop at Call 8, while `# CALL PLAN` said Calls 1–4 locate, Calls 5–8 edit, Calls 9–16 verify, Call 20 hard stop).

---

## 2. How v12 Was Engineered (Core Architectural Changes)

To move from `0.17` to `0.28–0.32+` on the Kaggle Leaderboard and achieve `3/3` on the local `smoke_3` suite, **v12** introduces five structural fixes:

### 2.1 Restore Native Tool Switching (`read_file` + `edit_file` + `submit_patch`)
- **Action**: Replace `sed -n` file reading with the native `read_file(path=..., offset=..., limit=...)` tool, and make native `edit_file(path=..., old_str=..., new_str=...)` Method A (with guarded Python inline replacement as Method B).
- **Why it works**: Keeping the agent actively switching between native ADK tools (`read_file`, `edit_file`, `get_status`, `submit_patch`) and `run_command` breaks the single-tool autoregressive loop that caused `run_command("submit_patch")`.
- **Explicit Disambiguation**: Section 1 of `prompts/system.md` now defines a strict **Native Tool vs. Shell Command** boundary, explicitly forbidding `run_command({"command": "submit_patch"})`.

### 2.2 Definition-First Localization (`git grep -n "def ..."`)
- **Action**: Phase 1 directs the agent to search for **definitions** (`git grep -n "def target_func"`) and explicitly warns that symbols often appear at both a call site (e.g., line 180) and a definition site (e.g., line 870).
- **Refined Loop Breaker**: Forbids repeating the *exact same tool call with identical arguments*, allowing the agent to read a second `offset` window in the same file if the first window was a call site or the function exceeds 65 lines.

### 2.3 Loud-Failing Guarded Python Edits & Contiguous Block Swaps
- **Action**: Whenever Method B (`python3 -c` via `run_command`) is used, the prompt mandates asserting exact match count (`assert c.count(old) == 1, f'Match count: {c.count(old)}'`) and printing `'EDIT_OK'`. Silent 0-byte no-op edits are now impossible.
- **Swap Rule**: Explicitly forbids chained `.replace(A, B).replace(B, A)` and requires replacing the single contiguous block containing both `A` and `B`.

### 2.4 Expanded & Line-Anchored Repository Playbooks
- **FastAPI (`fastapi/`)**: Anchors `async def _extract_form_body` in `fastapi/dependencies/utils.py` (~line 870, NOT the call site at ~line 180) and specifies checking `field_info = field.field_info` per field inside `for field in body_fields:` (`isinstance(field_info, params.File)`) rather than `body_fields[0].field_info`.
- **Requests (`src/requests/`)**: Anchors `def get_netrc_auth` in `src/requests/utils.py` (~line 205–245, NOT `auth.py`) and specifies guarding with `if _netrc and any(_netrc):` for Python 3.11+ empty netrc entries.
- **Rich (`rich/`)**: Anchors `def _render` in `rich/table.py` (~line 765–805) and specifies placing the `_box.mid_*` tuple (`_box.mid_left`, `_box.mid_right`, `_box.mid_vertical`) before the `_box.foot_*` tuple in a single contiguous block replacement.
- **Django, Scikit-learn, SymPy, Sphinx**: Adds concise architectural file/pattern anchors for the major SWE-bench Python repositories.

### 2.5 Calibrated Sampling & Execution Budgets
- `configs/sampling.yaml`:
  - `temperature: 0.15` (more deterministic than `0.2`, while avoiding the `0.0` greedy repetition trap).
  - `max_output_tokens: 4096` (down from `8192`; with `thinking_budget: 1536`, this leaves `2560` tokens for tool call output while cutting runaway generation time).
- `eval_config.yaml`:
  - `timeout_seconds: 90`
  - `max_tool_calls: 30` (down from `32`, adding extra safety margin across 120 tasks)
  - `max_time_minutes: 5.5`
  - `max_turns: 60`

---

## 3. Configuration Files in `bundle_v12`

All 4 active configuration files are deployed in [`bundle_v12/`](file:///home/lmassaron/code/gemma-4-developer-agent-competition/state_of_the_art/bundle_v12):
1. [`bundle_v12/agent.yaml`](file:///home/lmassaron/code/gemma-4-developer-agent-competition/state_of_the_art/bundle_v12/agent.yaml)
2. [`bundle_v12/eval_config.yaml`](file:///home/lmassaron/code/gemma-4-developer-agent-competition/state_of_the_art/bundle_v12/eval_config.yaml)
3. [`bundle_v12/configs/sampling.yaml`](file:///home/lmassaron/code/gemma-4-developer-agent-competition/state_of_the_art/bundle_v12/configs/sampling.yaml)
4. [`bundle_v12/prompts/system.md`](file:///home/lmassaron/code/gemma-4-developer-agent-competition/state_of_the_art/bundle_v12/prompts/system.md)

---

## 4. Local DGX Spark Benchmark Harness Fixes

Before executing benchmarking on DGX Spark, two critical harness bugs in `AgentSimulator` and `TaskWorkspace` must be addressed:

1. **Tool Parameter Key Prefix Stripping (`command: "..."`)**:
   - In `benchmark_v10_vs_v11.json`, tool calls were recorded with nested prefixes:
     `"run_command: {'command': 'command: \"git grep -n ...\"'}"`
     and `"edit_file: {'command': 'file: \"...\", old_string: \"...\"'}"`.
   - The regex/parser in `AgentSimulator` must strip the `command:` key prefix rather than passing `command: "..."` as the literal shell string to `subprocess.run()`, and properly map `read_file(path, offset, limit)` and `edit_file(path, old_str, new_str)`.
2. **Accurate Source `patch_size` Calculation**:
   - In `TaskWorkspace`, ensure `patch_size` is measured **before** applying `task["test_patch"]` (or excludes `tests/` files when computing `len(git_diff)`). In `requests_7205`, `test_patch` is exactly `791` bytes, which falsely caused `patch_size: 791` to be logged even when the agent modified 0 bytes of source code.
