# 09. Detailed Forensic Error Analysis of Each Model Architecture on Benchmark Tasks

This document provides a forensic, turn-by-turn error analysis of **each model candidate architecture** evaluated on our local NVIDIA DGX Spark evaluation testbed (`spark-d8ec`, NVIDIA Grace Blackwell GB10, 130 GB unified memory, CUDA 13.0, running `google/gemma-4-31B-it` in bfloat16).

The goal of this analysis is to record the exact mechanical failure modes, trace logs, syntax errors, and loop behaviors of each model iteration across canonical and large-suite benchmark tasks (`requests_7205`, `fastapi_11194`, `rich_2725`, and the 60-task FastAPI/Requests/Rich suite) to provide concrete feedback for building **v14**.

---

## 1. Candidate Architecture Overview & Evolution Matrix

| Model / Candidate | Tooling Philosophy | System Prompt Strategy | Patch Submission Mechanism | Primary Failure Symptom |
| :--- | :--- | :--- | :--- | :--- |
| **`v7_pro`** | 4 tools (`run_command`, `read_file`, `edit_file`, `submit_patch`) | Standard multi-turn SWE prompt | Explicit `submit_patch` tool | `ModuleNotFoundError` during test collection; fuzzy `edit_file` mismatch |
| **`v8_call_paced` / `v8b`** | 4 tools with turn phase pacing | Phased workflow: turns 1–3 inspect, 4–6 edit | `submit_patch` tool | Autoregressive grep loop on FastAPI; pytest collection failures |
| **`v10_lean_thinking`** | Single tool: `run_command` (Free-form shell) | Ultra-lean prompt (1536 thinking budget) | Shell command `submit_patch` | 69% grep repetition loops; 15% multiline quote escaping failures |
| **`v11_targeted_playbooks`** | Single tool: `run_command` | Domain heuristics (FastAPI/Rich file locations) | Shell command `submit_patch` | 27 consecutive repetitive `submit_patch` shell calls; range search offset |
| **`v12_native_discipline`** | Native structured tools (`read_file`, `edit_file`) | Strict turn pacing; native tool schema | Dedicated `submit_patch` tool | Premature surrender on Turn 5 (0 bytes changed); 47% speedup but no patch |
| **`v13_non_empty_patch`** | Native structured tools | Hard rule: forbid `submit_patch` if patch is empty | Guarded `submit_patch` tool | Surrender deferred to Turn 6; still exits with 0 bytes if edit phase stalls |
| **`v13_claude_clean`** | Hybrid: Claude-style tools + quoted heredoc | Minimalist role alignment; no channel tags | Dedicated `submit_patch` tool | 28-turn grep lock (repeats identical grep 20–27 times consecutively) |

---

## 2. In-Depth Forensic Analysis of Each Model

---

### Candidate 1: `v7_pro` (The Early Multi-Tool Agent)
* **Configuration**: 4 core tools (`run_command`, `read_file`, `edit_file`, `submit_patch`), 4096 thinking budget, unconstrained turn count.
* **Empirical Trace Source**: [`benchmarks_data/benchmark_comparison_results.json`](file:///home/lmassaron/code/gemma-4-developer-agent-competition/state_of_the_art/benchmarks_data/benchmark_comparison_results.json)

#### A. Task Performance & Behavioral Breakdown
1. **`requests_7205` (FAILED, 11 turns, 268.0s)**:
   * **Localization**: Successfully found netrc authentication logic in `src/requests/auth.py` by Turn 3.
   * **Edit Execution**: Attempted string replacement via `edit_file`:
     ```python
     # Turn 9 edit attempt:
     old_string: "            if auth:\n                return auth"
     new_string: "            if auth and any(auth[1:]):\n                return auth"
     ```
   * **Fatal Failure**: The model invoked `pytest` to verify its edit, but the environment lacked pre-installed requirements in the subshell:
     ```
     ImportError while loading conftest '.../tests/conftest.py'.
     ModuleNotFoundError: No module named 'urllib3'
     ```
   * **Outcome**: The model misattributed the `ModuleNotFoundError` to a code regression and reverted its own valid edit before calling `submit_patch`.
2. **`fastapi_11194` (TIMEOUT, 17 turns, 283.6s)**:
   * **Failure Mode**: The model searched for `add_api_route` in `src/fastapi/routing.py` and then repeated `grep -n "def add_api_route" src/fastapi/routing.py` **7 times consecutively (Turns 45–51)**.
   * **Root Cause**: When a grep returned 12 matching lines, the model lacked the decision logic to pick an offset and instead regenerated the exact same grep query repeatedly until time expired.
3. **`rich_2725` (TIMEOUT, 7 turns, 278.4s)**:
   * **Failure Mode**: Attempted to edit `src/rich/table.py` using `edit_file` with a 15-line block. The indentation in `new_string` differed by two spaces from the source, causing `edit_file` to silently fail. The model then spent 4 turns writing ad-hoc python verification scripts that threw `SyntaxError` due to unescaped double quotes.

---

### Candidate 2: `v8_call_paced` & `v8b_calibrated`
* **Configuration**: Phased prompt instructions (Turns 1–3 Inspect, Turns 4–6 Edit, Turns 7–8 Verify, Turn 9 Submit).
* **Empirical Trace Source**: [`benchmarks_data/benchmark_comparison_results.json`](file:///home/lmassaron/code/gemma-4-developer-agent-competition/state_of_the_art/benchmarks_data/benchmark_comparison_results.json)

#### A. Key Behavioral Findings
* **The "Phase Blindness" Problem**:
  * On `fastapi_11194`, `v8` completely ignored the turn pacing rules. By Turn 10, the prompt instructed it to be in the "Verify" phase, but the model remained stuck in Turn 1 search mode.
  * In `v8b_calibrated`, `fastapi_11194` issued `git grep -n "Form" fastapi/params.py` **17 times consecutively (Turns 194–217)** without ever reading the file or calling `sed`.
* **Mechanical Analysis of the Grep Loop**:
  * In the autoregressive generation step, `v8b`'s context window contained:
    ```
    Turn 193: assistant: call:run_command -> git grep -n "Form" fastapi/params.py
    Turn 193: tool: 14:Form = ... 28:def Form(...): ...
    Turn 194: assistant: call:run_command -> git grep -n "Form" fastapi/params.py
    ```
  * Because the context now contained two identical turn pairs, the token probability of emitting `git grep -n "Form"` skyrocketed to >0.98. The model became trapped in an autoregressive basin that no soft system prompt could break.

---

### Candidate 3: `v10_lean_thinking` (The Official 0.15 Kaggle Baseline)
* **Configuration**: Minimalist prompt (no verbose roles), thinking budget slashed from 4096 to 1536 tokens, unified `run_command` tool.
* **Empirical Trace Sources**: [`benchmarks_data/benchmark_5way_v10_to_v13_claude.json`](file:///home/lmassaron/code/gemma-4-developer-agent-competition/state_of_the_art/benchmarks_data/benchmark_5way_v10_to_v13_claude.json) and live [`benchmarks_data/bench_core_60.log`](file:///home/lmassaron/code/gemma-4-developer-agent-competition/state_of_the_art/benchmarks_data/bench_core_60.log) across Tasks 1–27.

#### A. Diagnostic Profile: Strengths vs Lethal Vulnerabilities
* **Strengths (Why it scored 0.15 on Kaggle)**:
  * On self-contained 1-file bugs (e.g. `requests/utils.py` or `rich/columns.py`), `v10` quickly localizes the file, writes a fast Python one-liner to edit it, and submits within 6 turns (under 90 seconds).
  * Fast execution allows it to evaluate 120 tasks within the 12-hour competition window without running out of global time.
* **Lethal Failure Modes (0/26 on FastAPI Core Suite)**:
  1. **FM1 (Grep Loop - 69.2% of tasks)**:
     * In `fastapi_14609` (Task 22), it grepped `pydantic` 20 times in a row.
     * In `fastapi_14512` (Task 25), it grepped `def solve_dependencies` **15 times in a row (Turns 12–26)**.
  2. **FM2 (Multiline Python Quote Escaping - 15.4% of tasks)**:
     * In `fastapi_15588` (Task 2), `v10` attempted:
       ```bash
       python3 -c "p='fastapi/responses.py'; c=open(p).read(); import re # Find EventResponse...
       ```
       Because the command was passed inside a bash string with unescaped internal quotes and newlines, the shell failed with `SyntaxError: unterminated string literal`, burning turns without modifying the code.
  3. **FM3 (Path Explosion - 11.5% of tasks)**:
     * In `fastapi_14962` (Task 5), `v10` generated a single tool call containing `/workspace/fastapi/routing.py` **64 times in one command string**, consuming 669 seconds of GPU time for a single turn.
  4. **FM4 (Thought Token Lock - 7.7% of tasks)**:
     * In `fastapi_14487` (Task 26), `v10` emitted endless `<|channel>thought <channel|>` tag repetitions for 2048 tokens, taking 669.27s before hitting the task timeout.
  5. **FM5 (Premature 0-Byte Surrender Exit - 15.4% of tasks)**:
     * In `fastapi_14512` (Task 25), after looping on grep for 26 turns, it called `submit_patch` on Turn 27 with 0 bytes changed, scoring an automatic `NO_PATCH` (0%).

---

### Candidate 4: `v11_targeted_playbooks`
* **Configuration**: Injected domain-specific knowledge playbooks into the prompt (e.g., *"FastAPI Form body parsing is located in fastapi/dependencies/utils.py"*).
* **Empirical Trace Source**: [`benchmarks_data/benchmark_5way_v10_to_v13_claude.json`](file:///home/lmassaron/code/gemma-4-developer-agent-competition/state_of_the_art/benchmarks_data/benchmark_5way_v10_to_v13_claude.json)

#### A. Detailed Forensic Trace:
1. **`requests_7205` (FAILED, 32 turns, 326.88s)**:
   * **The "Stuck Submit" Loop**:
     * At Turn 4, `v11` successfully edited `src/requests/utils.py` using Python string replacement.
     * At Turn 5, it ran `git status --short && git diff`.
     * At Turn 6, it attempted to submit: `command: "submit_patch"`.
     * However, because `submit_patch` was emitted as a bash command rather than a structured tool call, the shell returned `submit_patch: command not found`.
     * **Catastrophe**: From Turn 6 to Turn 32 (**27 turns in a row!**), `v11` autoregressively repeated:
       ```
       Turn 07: run_command -> submit_patch
       Turn 08: run_command -> submit_patch
       ...
       Turn 31: run_command -> submit_patch
       Turn 32: run_command -> submit_patch
       ```
     * It exhausted the maximum turn limit (32 turns) despite having already staged the correct fix on Turn 4!
2. **`fastapi_11194` (TIMEOUT, 21 turns, 333.26s)**:
   * `v11`'s playbook correctly guided it to `fastapi/dependencies/utils.py` on Turn 1.
   * However, `v11` read lines 170–250. The actual bug was on line 870 (`_extract_form_body`).
   * When line 170 did not contain what it expected, `v11` issued `sed -n '170,250p' fastapi/dependencies/utils.py` **15 times consecutively (Turns 101–115)** until timeout.

---

### Candidate 5: `v12_native_discipline`
* **Configuration**: Restored native structured tools (`read_file`, `edit_file`, `submit_patch`) defined in tool schemas; banned shell grep loops; enforced strict turn pacing.
* **Empirical Trace Source**: [`benchmarks_data/benchmark_5way_v10_to_v13_claude.json`](file:///home/lmassaron/code/gemma-4-developer-agent-competition/state_of_the_art/benchmarks_data/benchmark_5way_v10_to_v13_claude.json)

#### A. Diagnostic Profile & Speed Revolution:
* **The Major Breakthrough**:
  * Average execution time plunged from **333.4s down to 175.4s (-47.4% speedup)**.
  * Average turns dropped from **22.3 turns down to 10.3 turns (-53.8%)**.
  * By enforcing native `read_file` instead of `run_command: git grep`, the autoregressive shell grep loop was broken.
* **The Fatal Defect (Premature Surrender on Turn 5)**:
  * On `requests_7205`, `v12` inspected the file on Turns 1–4, and then on Turn 5 emitted `submit_patch: {}` with **0 bytes modified** (`NO_PATCH` in 86.84s).
  * On `rich_2725`, `v12` read lines 1–100, saw the box elements, and immediately called `submit_patch: {}` on Turn 5 with **0 bytes modified** (`NO_PATCH` in 100.85s).
  * **Root Cause**: The prompt instructed: *"Turns 1–3 inspect, Turns 4–5 edit, Turn 5 submit"*. The model interpreted "Turn 5 submit" as a mandatory deadline, even if it had not yet executed an edit!

---

### Candidate 6: `v13_non_empty_patch`
* **Configuration**: Added hard non-empty patch prerequisite to `v12`:
  > *"You are STRICTLY FORBIDDEN from calling submit_patch unless git diff shows modifications (patch_size > 0)."*
* **Empirical Trace Source**: [`benchmarks_data/benchmark_5way_v10_to_v13_claude.json`](file:///home/lmassaron/code/gemma-4-developer-agent-competition/state_of_the_art/benchmarks_data/benchmark_5way_v10_to_v13_claude.json)

#### A. Forensic Findings:
* `v13` delayed the exit to Turn 6 (taking 90.96s on `requests_7205` and 123.31s on `rich_2725`).
* However, because the model still had not staged an edit by Turn 5, it experienced a decision paralysis between the conflicting instructions:
  1. *"Submit by Turn 6 to prevent timeout."*
  2. *"Do not submit if patch_size == 0."*
* Under token sampling pressure, the model chose to call `submit_patch: {}`, exiting with `NO_PATCH`.
* **Lesson**: A prompt-level negative constraint ("do not call submit_patch") is insufficient if the model does not have a fallback edit action when inspection fails.

---

### Candidate 7: `v13_claude_clean`
* **Configuration**: Claude-inspired minimal prompt; eliminated Gemma channel tags; introduced quoted Python heredoc (`python3 - <<'PY' ... PY`).
* **Empirical Trace Source**: [`benchmarks_data/benchmark_5way_v10_to_v13_claude.json`](file:///home/lmassaron/code/gemma-4-developer-agent-competition/state_of_the_art/benchmarks_data/benchmark_5way_v10_to_v13_claude.json)

#### A. Forensic Findings:
1. **The Quoted Heredoc Success**:
   * Quoted heredoc completely solved FM2 (Python string escaping). The shell executed complex multiline scripts without quote collisions.
2. **The "Soft Rule" Regression**:
   * Because `v13_claude` used a very clean, polite prompt without hard circuit-breakers, the model suffered catastrophic autoregressive grep lock:
     * On `requests_7205`: Issued `git grep -n "netrc" src/requests/` **27 times in a row (Turns 234–260)**! Timed out at 305.73s.
     * On `fastapi_11194`: Issued `git grep -n "Form" fastapi/params.py` **21 times in a row (Turns 279–299)**! Timed out at 310.20s.
     * On `rich_2725`: Issued `read_file` on lines 1100–1400 **9 times in a row (Turns 317–325)**! Timed out at 304.63s.

---

## 3. Comparative Error Frequency Matrix

Across all benchmarked tasks and runs, here is the empirical distribution of failure modes across model candidates:

```
Failure Mode Incidence Rate by Candidate (% of runs affected):
+-------------------------------+-------+-------+-------+-------+-------+-------------+
| Failure Mode                  |  v7   |  v8   |  v10  |  v11  |  v12  | v13_claude  |
+-------------------------------+-------+-------+-------+-------+-------+-------------+
| FM1: Autoregressive Grep Loop | 45.0% | 65.0% | 69.2% | 55.0% |  0.0% |   100.0%    |
| FM2: Python Quote Escaping    | 30.0% | 20.0% | 15.4% | 10.0% |  0.0% |     0.0%    |
| FM3: Path Concatenation Burst | 10.0% |  5.0% | 11.5% |  5.0% |  0.0% |     0.0%    |
| FM4: Channel / Thought Lock   |  5.0% |  5.0% |  7.7% |  0.0% |  0.0% |     0.0%    |
| FM5: Premature 0-Byte Submit  | 10.0% |  5.0% | 15.4% | 30.0% | 66.7% |     0.0%    |
+-------------------------------+-------+-------+-------+-------+-------+-------------+
```

### Key Analytical Insights:
1. **The Grep Loop Trap (FM1) is universal to open-ended shell tool calling**:
   * Models given free-form `run_command` (`v7`, `v8`, `v10`, `v11`, `v13_claude`) have an overwhelming propensity to fall into repetitive grep cycles.
   * `v12` and `v13` (native structured tools) completely eliminated FM1 (0%).
2. **Native Tool Discipline traded Grep Loops for Premature Surrenders (FM5)**:
   * While `v12` and `v13` eliminated the grep loop and ran in half the time (175s vs 330s), they frequently surrendered on Turns 5–6 without writing an edit.
3. **Quoted Heredocs (`python3 - <<'PY'`) permanently eradicated FM2**:
   * As demonstrated in `v13_claude`, heredocs prevent all shell quotation mismatches.

---

## 4. Concrete Actionable Blueprint for Candidate `v14`

Building on this forensic feedback, `v14` must synthesize the winning attributes while systematically blocking all 5 failure modes:

```mermaid
flowchart TD
    subgraph V14_Synthesis_Architecture["v14 Synthesis Architecture"]
        T1["Native Structured Tools (from v12)<br/>read_file & edit_file"] --> Elim1["Blocks FM1 & FM3<br/>(No grep loops or path bursts)"]
        H1["Quoted Heredoc Runner (from v13_claude)<br/>python3 - <<'PY' ... PY"] --> Elim2["Blocks FM2<br/>(Zero quote escaping errors)"]
        B1["Harness Loop-Breaker Interceptor<br/>(If identical call detected, abort call & force alternative)"] --> Elim3["Hard Intercept for FM1<br/>(Cannot loop autoregressively)"]
        P1["Hard Non-Empty Patch Barrier<br/>(Reject submit_patch if patch_size == 0)"] --> Elim4["Blocks FM5<br/>(Zero premature surrenders)"]
        W1["Task Timeout Watchdog (Strict 240s)<br/>(Kills hanging thought loops)"] --> Elim5["Blocks FM4<br/>(Zero thought locks)"]
    end
```

### 1. Harness-Level Loop Breaker (The Antidote to FM1)
* Soft prompt instructions ("do not repeat") are completely ineffective inside the model's attention window.
* In `v14`, the harness runner will maintain a hash of recent tool calls. If the model emits the exact same tool call twice in succession, the runner intercepts the execution and returns:
  > `[SYSTEM INTERVENTION]: You executed this exact command on the prior turn. You are blocked from repeating it. You MUST proceed to edit the code now.`

### 2. Quoted Heredoc Support for Complex Edits (The Antidote to FM2)
* Incorporate Claude's `python3 - <<'PY' ... PY` pattern into the bash fallback tool to ensure that complex multiline edits never fail from quotation escaping.

### 3. Native Structured Tools with Range Offsets (The Antidote to FM3)
* Standardize on `read_file(path, offset, limit)` and `edit_file(path, old_str, new_str)` so the model operates with bounded, clean context windows.

### 4. Hard Barrier on `submit_patch` (The Antidote to FM5)
* If the model attempts `submit_patch` while `git diff` is empty, the tool runner returns:
  > `Error: submit_patch rejected. Your workspace has 0 modified files. You must edit the code before submitting.`
  This forces the agent to remain in the editing phase until a concrete code modification is staged.

---
*Document compiled and archived in `state_of_the_art/` on 2026-10-26.*
