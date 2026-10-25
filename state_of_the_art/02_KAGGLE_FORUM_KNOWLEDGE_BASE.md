# Kaggle Forum Knowledge Base & Competitive Intelligence

> **Synthesized from all 258 Kaggle discussion threads** across Pages 1–13 (sorted by votes), along with code audits of top public notebooks (`flexonafft`, `honghanhhh`, `jvnhonglei`, `dmitriigluzdov`, `samvalladares`, `lavinwins`).

---

## 1. Top Performing Public Notebooks: Architecture & Score Breakdown

| Notebook Title | Author | Medals | Declared LB | Core Architecture & Key Mechanisms |
| :--- | :--- | :---: | :---: | :--- |
| **`gemma-4-baseline-lb-0-24`** | `flexonafft` / community | Gold | **0.24** | **Targeted Prompt Refinement on Single Agent**: Strict call pacing, parallel sync/async implementation rules, and protected test paths (`pytest.ini`, `conftest.py`). Declares graph tools in YAML but explicitly forbids their use in prompt ("They return nothing useful"). |
| **`gemma-4-budget-aware-coding-agent-v2`** | `flexonafft` | Silver | **0.22** | **Budget-Aware Operating Profile**: Enforces turn-12 edit deadline, bounded independent searches, and `git status --short && git diff` verification. Guarded unique-match replacement. |
| **`gemma-4-superagent-reproduction`** | `jvnhonglei` | Silver | **0.20** | **4-Phase Strict Sequential Execution**: Calls 1–8 locate/read, Calls 9–12 edit, Calls 13–20 verify, Call 20+ submit. Hard cutoff on exploration at turn 12. |
| **`gemma-4-agent-budget-fit-single-agent`** | `samvalladares` | Bronze | **0.18** | **Calibrated Budget Schedule**: First to reduce tool calls from 40–100 down to 28 calls / 8 minutes based on 129 local test runs. |
| **`gemma-4-autonomous-developer-agent`** | `lavinwins` | Bronze | **0.18** | **Scratch Hygiene**: Strict scratch file containment (`/tmp` only) to prevent untracked files from polluting the evaluation patch. |
| **`gemma-4: public-agent-ideas-and-evidence`** | `dmitriigluzdov` | Bronze | **0.18** | **Empirical Taxonomy of 187 Notebooks & 74 Mechanisms**: Proved prompt and tools stay within noise, while time per task is the primary binding constraint. |

---

## 2. The 0.20 $\to$ 0.22 $\to$ 0.24 Progression

A byte-by-byte diff of `gemma-4-baseline-lb-0-24` against the 0.20 and 0.22 baselines revealed that:
* `agent.yaml`: **Byte-identical**
* `configs/sampling.yaml`: **Byte-identical**
* `eval_config.yaml`: **Byte-identical**
* Base model: **Byte-identical** (`gemma-4-31b-it-qat-w4a16-ct`)
* LoRA adapters: **None** (all ran zero-shot base models!)

Every single point of difference between 0.20 and 0.24 was driven by three procedural directives in `prompts/system.md`:

1. **The Parallel Implementation Law**:
   * Python libraries frequently implement twin classes (e.g. `Client` vs `AsyncClient` in `httpx`, synchronous route handlers vs `async def` in `fastapi`, and dual export in `__all__` and `__init__.py`).
   * Directive: *"Apply a change to every parallel implementation (sync/async, Client/AsyncClient, backends, overloads, `__init__`/`__all__` re-exports). Hidden tests often exercise both."*
2. **Protecting Test Infrastructure**:
   * Naive agents often modify `pytest.ini` or `conftest.py` to bypass failing tests.
   * Directive: *"Never create, modify or delete test files (`test_*.py`, `*_test.py`, anything under `tests/`, `pytest.ini`, `conftest.py`). This is an automatic failure."*
3. **The Untracked File Trap (`git status --short`)**:
   * `git diff` alone hides untracked scratch files, but `submit_patch` uses `git add -N . && git diff HEAD`.
   * Directive: *"Always inspect `git status --short && git diff` before calling `submit_patch`."*

---

## 3. The 4 Fatal Traps Reported Across 258 Forum Discussions

### Trap 1: The 12-Hour "Notebook Timeout" Epidemic (Topics #12, #31, #75, #173, #182)
* **Root Cause**: Setting `max_time_minutes: 8` or `10`. When an agent struggles on 50+ tasks, total execution breaches the 720-minute notebook cutoff.
* **Solution**: A hard cap of **5.5 minutes (330s)** and **32 tool calls** guarantees execution in $\le 11.0$ hours with an ample safety margin.

### Trap 2: The Zero-Patch Scourge (`patch_size == 0`) (Topics #101, #149, #305)
* **Root Cause**: 38% of tasks in default starter kits never edit a source file. When `edit_file` fails due to whitespace mismatches, naive agents get stuck in search loops until time expires.
* **Solution**: The **5-Turn Edit Mandate** and **Call-8 Emergency Patch Rule**: force the agent to apply its best-guess edit by Turn 5 (and hard stop at Turn 8). An imperfect edit scores infinitely better than an unedited repository.

### Trap 3: Context Compaction Thrashing (Topics #71, #85, #93)
* **Root Cause**: Dumping full test suite outputs or thousands of lines of source code triggers ADK's 32k context compaction. Compaction strips earlier tool call history, causing the model to forget that it already ran a search, inducing infinite repetition loops.
* **Solution**: Limit test outputs to `tail -40`, name one test file with `-k`, and enforce a hard exploration cutoff by Turn 12.

### Trap 4: Pre-Existing Environment Failures (Topics #18, #30, #35, #78, #108, #170)
* **Root Cause**: In offline evaluation sandboxes, optional dependencies (`starlette`, `python-multipart`, `typing_inspection`, `inline_snapshot`) are missing. 0 out of 64 reference patches pass if the agent runs full pytest suites.
* **Solution**: Explicit offline immunity prompt directive: *"Failures that come from missing fixtures, missing optional dependencies, or import errors unrelated to your change are pre-existing. Never attempt to repair tests or the environment. Verify syntax with `python3 -m py_compile` and submit."*

---

## 4. Quantitative Ablation Data

### A. Sampling & Thinking Ablation (Ayush Thakur, CoreWeave 32× B40 GPUs)
*Evaluated across 129 SWE-bench tasks with `gemma-4-31b-it-qat-w4a16-ct`.*

| Configuration | Setting | Resolved / 129 | Resolution Rate | Observations |
| :--- | :--- | :---: | :---: | :--- |
| **Baseline** | Temp 1.0, Top-p 0.95, Budget 4096 | **51 / 129** | **39.53%** | Reference standard |
| **Temperature** | Temp 0.2 / 0.4 | **51 / 129** | **39.53%** | Ties Temp 1.0; more deterministic tool syntax |
| | Temp 0.0 (Greedy) | 47 / 129 | 36.43% | Susceptible to repetitive search loops |
| **Thinking Toggle** | **Thinking Disabled** | **23 / 129** | **17.83%** | **Catastrophic -55% collapse**. Thinking is mandatory. |
| **Thought Summaries**| Disabled | 35 / 129 | 27.13% | Loses multi-turn planning cohesion |
| **Thinking Budget** | Budget 1,536 – 2,048 tokens | **50 / 129** | **38.76%** | **Optimal speed/quality balance** (-45% latency) |
| | Budget 6,144 tokens | 48 / 129 | 37.21% | Slight regression; model over-thinks simple fixes |

### B. Teacher Comparison: Is Gemma 4 31B the Limiting Factor? (Ruijie Zhu)
*Evaluated across 124 public SWE-bench tasks.*

| Model | Solved / 124 | Median Minutes / Task |
| :--- | :---: | :---: |
| **Gemma 4 31B** | 54 / 124 (43.5%) | 5.2 min |
| **DeepSeek V4 Pro** | 58 / 124 (46.8%) | 3.5 min |
| **Qwen3.6-27B** | 59 / 124 (47.9%) | 3.9 min |

*Takeaway*: Frontier models solve only 4–5 more tasks than Gemma 4 31B. Performance is governed by harness limits and search discipline, not raw model IQ.

---

## 5. Why LoRA Fine-Tuning Failed in this Competition

Multiple participants (including our own `train_31b.log` and `train_12b.log`) attempted SFT fine-tuning, but all public LoRA submissions lagged behind zero-shot base models:
1. **vLLM Layer Bug (Topic #39)**: In vLLM 0.19.1, Gemma 4 decoder layers are registered twice, causing LoRA adapters to be silently dropped under high load.
2. **KV Cache Collapse (Topic #57)**: When LoRA is loaded on 4× L4 GPUs, the memory footprint shrinks the KV cache to ~7.6k tokens. Multi-turn rollouts quickly overflow and hang the inference server.
3. **Conversational Drift**: Off-policy trajectories teach the model to chat rather than call tools deterministically.
