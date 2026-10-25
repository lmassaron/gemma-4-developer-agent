# 03. Current Solution: Architecture, v10 vs v11 Comparison & Engineering Directives

This document provides a comprehensive technical breakdown of our active production configurations: **v10 (Lean Thinking & Decisive)** and **v11 (Targeted Domain Playbooks & Emergency Patching)**. Both configurations are fully mirrored in this directory under [`bundle_v10/`](file:///home/lmassaron/code/gemma-4-developer-agent-competition/state_of_the_art/bundle_v10) and [`bundle_v11/`](file:///home/lmassaron/code/gemma-4-developer-agent-competition/state_of_the_art/bundle_v11).

---

## 1. High-Level Architectural Comparison: v10 vs v11

| Dimension | v10 ("Lean Thinking & Decisive") | v11 ("Targeted Playbooks & Emergency Patching") | Architectural Rationale |
| :--- | :--- | :--- | :--- |
| **Thinking Budget** | `1536` tokens | `1536` tokens | Decreased from `4096` (v9). Halves per-turn decoding latency on Gemma 4 31B without degrading logical deduction. |
| **Max Time Budget** | `5.5` minutes / task | `5.5` minutes / task | $120 \text{ tasks} \times 5.5 \text{ min} = 11.0 \text{ hours}$. Reserves a strict 60-minute buffer under the 12-hour Kaggle ceiling. |
| **Max Tool Calls** | `32` tool calls | `32` tool calls | Prevents runaway agents from burning time after wandering past turn 20. |
| **System Prompt Size** | ~1,850 tokens | ~2,400 tokens | Added targeted architectural domain playbooks for frequent repository archetypes (FastAPI, Requests, Rich). |
| **Edit Phase Mandate** | Strict Turn 5 deadline | Strict Turn 5 deadline | Forbids read-only exploration after Turn 5. Agent must commit edits by Turn 5. |
| **Emergency Patch Fallback**| None (relied on agent discipline) | **Mandatory Turn 8 Emergency Trigger** | If no patch is staged by Turn 8, the agent must immediately synthesize a minimal patch and call `submit_patch`. |
| **Tool Execution Strategy** | Standard shell & edit tools | **Python 1-line replace over `edit_file`** | `edit_file` repeatedly failed due to indentation/whitespace mismatches. Python scripts guarantee atomic, exact substring replacements. |
| **Domain Playbooks** | Generic guidance | **Specific Repository Playbooks** | Added precise failure modes for `fastapi` (`_extract_form_body`), `requests` (`unquote`), and `rich` (`Table` measurements). |

---

## 2. Configuration Files Breakdown

### 2.1 `eval_config.yaml`
Location: [`bundle_v11/eval_config.yaml`](file:///home/lmassaron/code/gemma-4-developer-agent-competition/state_of_the_art/bundle_v11/eval_config.yaml)

```yaml
max_time_minutes: 5.5
max_tool_calls: 32
timeout_per_command_seconds: 45
```

- **`max_time_minutes: 5.5`**: Ensures the agent is hard-terminated if a task hangs on an infinite loop or expensive test suite.
- **`max_tool_calls: 32`**: Caps cumulative context size. At ~1,000 tokens per tool response, 32 calls keep context under 32,768 tokens, avoiding the quadratic slowdown of full-attention decoding.
- **`timeout_per_command_seconds: 45`**: Guarantees test suites (e.g., `pytest test/`) do not freeze when waiting on unmocked network sockets or deadlocks.

### 2.2 `configs/sampling.yaml`
Location: [`bundle_v11/configs/sampling.yaml`](file:///home/lmassaron/code/gemma-4-developer-agent-competition/state_of_the_art/bundle_v11/configs/sampling.yaml)

```yaml
temperature: 0.1
top_p: 0.95
max_tokens: 3072
thinking_budget: 1536
```

- **`temperature: 0.1`**: Keeps sampling virtually greedy for reproducible, syntax-valid code generation and exact tool calling.
- **`thinking_budget: 1536`**: Gemma 4 31B outputs an internal reasoning block enclosed in `<thought>...</thought>`. By default, it generated 4,096 tokens of chain-of-thought per turn. Restricting this to 1,536 tokens forces the model to concisely formulate hypotheses and action steps, cutting turn latency from ~28s to ~12s.
- **`max_tokens: 3072`**: Provides sufficient remaining generation budget (1,536 tokens) for generating diffs or multiline python scripts in a single tool call.

### 2.3 `agent.yaml`
Location: [`bundle_v11/agent.yaml`](file:///home/lmassaron/code/gemma-4-developer-agent-competition/state_of_the_art/bundle_v11/agent.yaml)

```yaml
system_prompt_path: prompts/system.md
sampling_config_path: configs/sampling.yaml
tools:
  - run_command
  - read_file
  - edit_file
  - submit_patch
```
*Note: Advanced graph/LSP tools (`symbol_definition`, `list_directory_symbols`) are deliberately omitted due to high failure rates and parsing overhead.*

---

## 3. System Prompt Engineering (`prompts/system.md`)

The complete v11 prompt is available at [`bundle_v11/prompts/system.md`](file:///home/lmassaron/code/gemma-4-developer-agent-competition/state_of_the_art/bundle_v11/prompts/system.md). Key innovations include:

### 3.1 Strict Turn Pacing and Phasing
```markdown
## Turn Budget & Execution Phases:
- Turns 1-2: Locate exact bug files and reproduction snippet.
- Turns 3-4: Verify bug with a standalone test command (`python -c "..."` or `pytest -k ...`).
- Turn 5: MANDATORY EDIT PHASE. Apply code changes. Do not continue exploring past Turn 5.
- Turns 6-7: Run the verification test and run `git diff` to ensure cleanliness.
- Turn 8: MANDATORY SUBMISSION. Call `submit_patch`.
- EMERGENCY TRIGGER: If you reach Turn 8 without submitting, immediately call `submit_patch` on your current staged changes.
```

### 3.2 Python 1-Line Surgical Edits vs `edit_file`
A major finding in our DGX Spark benchmarks was that Gemma 4 31B frequently fails with `edit_file` when replacing blocks in Python code due to:
1. Slight whitespace/indentation mismatches between `old_str` and the file.
2. Inability to uniquely match common syntax patterns across lines.

v11 instructs the model:
```markdown
## Editing Code:
Prefer atomic Python replacement scripts over multiline edit_file when indentation is delicate:
python3 -c "
with open('path/to/file.py', 'r') as f:
    content = f.read()
content = content.replace('OLD_SNIPPET', 'NEW_SNIPPET', 1)
with open('path/to/file.py', 'w') as f:
    f.write(content)
"
Always verify edits immediately with:
git diff path/to/file.py
```

### 3.3 Domain-Specific Repository Playbooks
Kaggle's test set contains recurrent architecture patterns. v11 explicitly equips Gemma 4 with domain rules:

1. **FastAPI / Pydantic Form Body Playbook**:
   - Issue: Nested models or form data parsing errors when receiving multiple fields or extra form parameters.
   - Core file: `fastapi/dependencies/utils.py`.
   - Core function: `_extract_form_body(field, values)`.
   - Rule: Ensure dict lookups handle `FormData` objects and fallback gracefully when keys are omitted or multivalued.

2. **Requests / HTTP Header & URL Encoding Playbook**:
   - Issue: Double unquoting of URLs, query parameters stripping, or improper handling of custom headers in redirects.
   - Core file: `src/requests/utils.py` or `requests/sessions.py`.
   - Core function: `unquote_unreserved` and `requote_uri`.
   - Rule: Never modify raw URL paths if already percent-encoded; preserve query strings across redirects.

3. **Rich / Console & Table Measurement Playbook**:
   - Issue: Crash or incorrect padding calculation when rendering empty tables, terminal width overflows, or zero-width ANSI strings.
   - Core file: `rich/table.py` or `rich/console.py`.
   - Core function: `Table.__rich_measure__` and column size allocation.
   - Rule: Guard against division by zero when `max_width == 0` or total column ratio is 0.

---

## 4. Packaging and Validation Pipeline

The submission bundle is packaged and strictly validated using [`scripts/package_and_validate.py`](file:///home/lmassaron/code/gemma-4-developer-agent-competition/state_of_the_art/scripts/package_and_validate.py):
1. Verifies root directory structure inside zip (`agent.yaml` must be at root).
2. Validates YAML schemas against ADK expectations.
3. Checks total compressed size ($\le 50 \text{ MB}$).
4. Verifies presence of all referenced prompt files and configs.
5. Computes SHA-256 integrity hash.
