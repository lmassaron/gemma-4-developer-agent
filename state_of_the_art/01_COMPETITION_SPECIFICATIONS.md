# Competition Specifications & Evaluation Harness Contract

## 1. Official Evaluation Environment

* **Platform**: Kaggle Code Competition (Air-Gapped, No Internet Access).
* **Hardware Profile**: 4× NVIDIA L4 GPU instance (24 GB VRAM each = 96 GB total GPU VRAM) running under a hardened Linux container.
* **Global Time Limit**: **12.0 hours (720 minutes)** for the entire submission run.
* **Evaluation Suite**: ~120 sequential SWE-bench tasks:
  * Public partition: **58 tasks** (determines the Public Leaderboard score).
  * Private partition: **~62 tasks** (determines final competition standing).
  * The evaluation proceeds task-by-task in a single continuous process.

---

## 2. The Critical Time Arithmetic

Because tasks are processed sequentially:
$$\text{Average Available Time Per Task} = \frac{720\text{ minutes}}{120\text{ tasks}} = 360\text{ seconds (6.0 minutes)}$$

* **Container Overhead**: Snapshot unpacking, Git resets, and patch extraction consume 30–45s per task.
* **The Failure Mode of Naive Configurations**:
  * Starter kits set `max_time_minutes: 8` and `max_tool_calls: 80`.
  * If an agent struggles on 50 tasks, consuming the full 8 minutes on each:
    $$50 \times 8\text{ min} + 50 \times 4\text{ min} = 600\text{ min (10 hours)}$$
    By task ~75–80, the global 12-hour timer expires, producing a **Notebook Timeout** and invalidating all subsequent tasks.
* **The Winning Constraint**:
  * `max_time_minutes: 5.5`
  * `max_tool_calls: 32`
  * Under this profile, worst-case runtime across 120 tasks is $120 \times 5.5\text{m} = 660\text{ minutes (11.0 hours)}$, leaving an exact **1.0-hour safety buffer** against Kaggle timeouts.

---

## 3. ADK Harness Bundle Contract (`submission.zip`)

The evaluation harness uses Google's Agent Development Kit (`adk-submission`). The root of `submission.zip` must contain:

```text
submission.zip
├── agent.yaml                   # Root agent definition
├── eval_config.yaml             # Per-task execution budget
├── configs/
│   └── sampling.yaml            # Generation & thinking hyperparameters
└── prompts/
    └── system.md                # System prompt text
```

### File 1: `agent.yaml`
```yaml
name: swe_agent
model: gemma-4-31b-it-qat-w4a16-ct
instruction: !include prompts/system.md
tools:
  - run_command
  - read_file
  - edit_file
  - write_file
  - get_status
  - submit_patch
  - get_code_neighbors
  - search_similar_code
  - get_code_subgraph
generate_content_config: !include configs/sampling.yaml
```

### File 2: `configs/sampling.yaml`
```yaml
temperature: 0.2
top_p: 0.95
max_output_tokens: 8192
thinking_config:
  thinking_budget: 1536
  include_thoughts: true
```
* `thinking_budget: 1536`: Proven optimal on DGX Spark. Reduces decoding latency by ~50% compared to 4096 tokens while retaining complete deductive reasoning.

### File 3: `eval_config.yaml`
```yaml
evaluation:
  timeout_seconds: 120
  max_tool_calls: 32
  max_time_minutes: 5.5
  max_turns: 80
```

---

## 4. ADK Toolset: Capabilities & Traps

The harness exposes 9 tools to the model:

| Tool | Signature | Function | Known Trap / Failure Mode |
| :--- | :--- | :--- | :--- |
| `run_command` | `{"command": "string"}` | Executes shell command in `/workspace` | **5,000-char truncation**: pytest outputs beyond 5,000 chars are cut off; the bottom failure summary is lost. Must use `tail -40`. |
| `read_file` | `{"path": "str", "offset": int, "limit": int}` | Reads line-numbered file slice | Dumping whole files fills context and triggers 32k context compaction. |
| `edit_file` | `{"path": "str", "old_str": "str", "new_str": "str"}` | Exact string replacement | **Whitespace / Indentation Mismatch**: If `old_str` has 1 space mismatch or tab difference, it fails. vLLM parser also mangles `"""` docstrings. |
| `write_file` | `{"path": "str", "content": "str"}` | Creates or overwrites file | Overwriting entire files causes massive unintended regressions. |
| `get_status` | `{}` | Runs `git status --short` and `git diff --stat` | Essential for checking untracked files before submission. |
| `submit_patch` | `{}` | **Terminal Tool**: Signals completion | **Must be called as a native tool, NOT via `run_command("submit_patch")`**. Calling it inside `run_command` produces `command not found`. |
| `search_similar_code` | `{"query": "str"}` | Semantic code search | **Context Bomb (Topic #114)**: Dumps near-collapsed embeddings and thousands of tokens into context; consistently returns empty or noisy results. **Must be declared in YAML but forbidden in prompt.** |
| `get_code_neighbors` | `{"symbol": "str"}` | AST code graph neighbor lookup | Often empty; misses async functions entirely (Topic #25). |
| `get_code_subgraph` | `{"symbol": "str"}` | AST dependency graph | Harms performance in ablations (-3.8% solve rate, Ayush Thakur). |

---

## 5. Offline Air-Gapped Environment & Test Sandbox Traps

1. **Missing Optional Dependencies in Test Sandbox**:
   * Open-source repositories like FastAPI and Requests contain optional dependencies (`starlette`, `urllib3`, `python-multipart`, `inline_snapshot`, `redis`).
   * In Kaggle's evaluation container (and local sandboxes), these packages are frequently missing or version-mismatched.
   * If the agent runs `pytest tests/`, tests fail with `ModuleNotFoundError`.
   * **Rule**: Failures due to `ModuleNotFoundError` or missing fixtures are pre-existing environment issues. The agent must verify syntax with `python3 -m py_compile <file>` and submit immediately without touching test files.
2. **The Untracked File Trap (`git add -N .`)**:
   * The evaluation harness extracts the patch using:
     ```bash
     git add -N . && git diff --binary HEAD
     ```
   * Any scratch file created in `/workspace/` (e.g. `/workspace/repro.py`) is staged into the graded patch!
   * Unexpected test files or reproduction scripts break repository linters and cause automatic evaluation failure.
   * **Rule**: Scratch files belong strictly in `/tmp/`.
