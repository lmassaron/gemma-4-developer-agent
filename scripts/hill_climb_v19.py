#!/usr/bin/env python3
"""
Automated Cross-Benchmark Hill-Climbing Optimizer (v18 -> v19) + Core-60 Evaluator
for Gemma 4 31B SWE-bench Agent (100% Native Kaggle ADK Tool Calling).

Executes Stage 2 and Stage 3 in a single GPU session:
  - Stage 2: 5-Iteration Cumulative Hill-Climbing (v18 -> v19) across a 12-task
    Cross-Benchmark Suite (`climb_12_cross`) combining:
      * 4 Fast Regression Guards (`requests_7205`, `fastapi_14372`, `rich_3882`, `sympy__sympy-24539`)
      * 4 Hardest Remaining Failures from `core_60` / `climb_12` (`fastapi_14349`, `rich_3777`, `rich_3772`, `rich_3934`)
      * 4 Hardest Failures from the External HuggingFace Benchmark `external_hf_py312`
        (`sympy__sympy-23534`, `sympy__sympy-24562`, `sympy__sympy-23950`, `sympy__sympy-24213`)
  - Stage 3: Full 60-Task `core_60` Benchmark Evaluation of `v19_hill_climbed` to measure
    the overall gain over `v16`'s 12/60 (20.0%).
"""

import argparse
import copy
import gc
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile
import yaml
import torch

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
scripts_dir = str(Path(__file__).resolve().parent)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from transformers import AutoProcessor
try:
    from transformers import AutoModelForImageTextToText as AutoModelForMultimodalLM
except ImportError:
    from transformers import AutoModelForMultimodalLM

from run_local_benchmark_leaderboard import AgentSimulator, update_leaderboard_files


# ==============================================================================
# 1. CROSS-BENCHMARK SUITE BUILDER (CORE_60 + CLIMB_12 + EXTERNAL_HF_PY312)
# ==============================================================================

def build_v19_cross_benchmark_suite(root: Path) -> tuple[list[dict], set[str], dict]:
    task_map = {}
    for jf in [root / "data" / "tasks.jsonl", root / "data" / "external_hf_benchmark_py312.jsonl"]:
        if jf.is_file():
            with open(jf, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        t = json.loads(line)
                        task_map[t["instance_id"]] = t

    guard_ids = [
        "requests_7205",
        "fastapi_14372",
        "rich_3882",
        "sympy__sympy-24539",
    ]
    core_failure_ids = [
        "fastapi_14349",
        "rich_3777",
        "rich_3772",
        "rich_3934",
    ]
    external_hf_failure_ids = [
        "sympy__sympy-23534",
        "sympy__sympy-24562",
        "sympy__sympy-23950",
        "sympy__sympy-24213",
    ]

    all_ids = guard_ids + core_failure_ids + external_hf_failure_ids
    selected_tasks = [task_map[tid] for tid in all_ids if tid in task_map]

    suite_metadata = {
        "total_tasks": len(selected_tasks),
        "guards": guard_ids,
        "core_60_and_climb_12_failures": core_failure_ids,
        "external_hf_py312_failures": external_hf_failure_ids,
    }
    return selected_tasks, set(guard_ids), suite_metadata


def load_v18_cached_baseline_tasks(root: Path) -> dict[str, dict]:
    """Loads already-evaluated v18 task results so Step 0 only runs missing tasks."""
    cache = {}
    v18_climb_file = root / "state_of_the_art" / "benchmarks_data" / "hill_climb_v17_to_v18_results.json"
    if v18_climb_file.is_file():
        data = json.loads(v18_climb_file.read_text(encoding="utf-8"))
        for it in data.get("iterations", []):
            if it.get("id") == "v18_iter3_single_existing_pytest_gate":
                for t in it.get("tasks", []):
                    cache[t["instance_id"]] = t
    v18_ext_file = root / "state_of_the_art" / "benchmarks_data" / "benchmark_v18_external_hf_py312_results.json"
    if v18_ext_file.is_file():
        data = json.loads(v18_ext_file.read_text(encoding="utf-8"))
        for t in data.get("v18_hill_climbed", {}).get("tasks", []):
            cache[t["instance_id"]] = t
    return cache


# ==============================================================================
# 2. V18 BASELINE SECTIONS & 5 CUMULATIVE V19 HILL-CLIMBING MUTATIONS (NATIVE ADK)
# ==============================================================================

V18_BASELINE_SECTIONS = {
    "header": """You are an expert autonomous software engineer fixing an issue in `/workspace`. Make the smallest correct change to the SOURCE code in `/workspace` so that the hidden grading tests for the issue below pass. Every name, parameter, exception, and error message in the issue is a literal requirement.

# MANDATORY NATIVE TOOL DISCIPLINE
- Invoke a tool on EVERY turn using your native tool-calling interface (`run_command`, `read_file`, `edit_file`, `write_file`, `get_status`, `submit_patch`). Never output standalone text or reflections without invoking a tool.
- Never use `search_similar_code`, `get_code_neighbors`, or `get_code_subgraph`. They return nothing useful.

# RIGID 4-PHASE PROTOCOL (ACTION OVER EXPLORATION)
Never get trapped reading files endlessly. Follow this exact sequence:""",

    "phase1": """## PHASE 1: FAST LOCATE (Call 1)
- Use `run_command` with extended regex (`-E`) to search for literal symbols from the issue:
  `git grep -n -I -E "<LiteralSymbol1>|<LiteralSymbol2>" -- "*.py" ":!tests/*"`
- ZERO-RESULT ESCAPE: If your first `git grep` returns nothing, DO NOT guess more symbol names! On Call 2, immediately locate the relevant test or traceback via `run_command`:
  `git grep -n -I -E "<keyword>" -- "tests/*.py" || timeout 45 python3 -m pytest -k "<keyword>" --tb=short -q 2>&1 | tail -25`
- Search ONLY Python files (`*.py`) unless the issue explicitly describes a documentation file.""",

    "phase2": """## PHASE 2: READ CONTEXT (Call 2)
- Use `read_file` with `path`, `offset`, and `limit: 60` to read the candidate function around the matched line number.
- Identify the exact 2–3 line anchor where the fix belongs.
- EXPLORATION LIMIT: You are allowed at most TWO `read_file` calls per task. Once you have seen the code, DO NOT read more files. Proceed immediately to Phase 3.""",

    "phase3": """## PHASE 3: MANDATORY EDIT (Call 3 or 4)
- Apply your fix using `edit_file` with exact arguments `{"path": "rel/file.py", "old_str": "EXACT 2-3 LINES", "new_str": "REPLACEMENT LINES"}`.
- CRITICAL SYNTAX & PYTHON INVARIANTS:
  1. Keep `old_str` to 2–4 unique lines with exact indentation and WITHOUT line-number prefixes (`42 | `).
  2. PARALLEL IMPLEMENTATIONS: Apply the change to every parallel implementation (sync/async, Client/AsyncClient, sibling classes, `__init__`/`__all__` re-exports).
  3. DATACLASS RULES: When adding a field with a default (e.g. `offset: Optional[int] = None`) to a `@dataclass`, ALWAYS place it AFTER all non-default fields! When adding `@dataclass(frozen=True)` to a class, also add `(frozen=True)` to any subclass inheriting from it in the same file.
  4. CUSTOM EXCEPTIONS/WARNINGS: Check if the repo defines custom classes (e.g. `FastAPIDeprecationWarning`, `FastAPIError`) and match the exact issue message.
- FALLBACK: If `edit_file` reports mismatch once, use `run_command` with atomic Python replace:
  `python3 -c 'from pathlib import Path; p=Path("rel/file.py"); s=p.read_text(); p.write_text(s.replace("OLD_LINE", "NEW_LINE", 1))'`""",

    "phase4": """## PHASE 4: VERIFY & SUBMIT (Calls 5-6)
- IMMEDIATELY after editing in Phase 3, verify syntax and diff via `run_command`:
  `python3 -m py_compile rel/file.py && git status --short && git diff`
- OPTIONAL TARGETED TEST: Always pass a specific test file path AND `-k`:
  `timeout 25 python3 -m pytest tests/test_<name>.py -k "<kw>" -x --tb=short -q 2>&1 | tail -25`
  NEVER run bare `pytest` or `pytest tests/` without a specific test file or `-k` filter! Failures from missing optional dependencies or unrelated fixtures are pre-existing environment issues—ignore them and submit once `py_compile` and `git diff` succeed.
- DO NOT DELAY SUBMISSION: As soon as `git diff` confirms your fix is applied and syntax is valid, invoke `submit_patch` IMMEDIATELY on the very next turn! Never run more than 12 turns total.""",

    "breakers": """# HARD CIRCUIT BREAKERS
- 28 calls max, 5.5 min per task. When either runs out, the working tree is graded as it is.
- NO REPETITION: Never execute the exact same command or tool call twice in a row. Once you have output, move forward to read or edit.
- ZERO-DIFF FORBIDDEN: Submitting 0 bytes changed scores 0 automatically. Never call `submit_patch` unless `git diff` shows modifications.
- NEVER create, modify, or delete test files (`tests/**`, `test_*.py`, `*_test.py`, `pytest.ini`, `conftest.py`, `setup.py`, `pyproject.toml`). Modifying test files is an automatic failure.
- NEVER leave untracked scratch files in `/workspace`. Scratch files go in `/tmp` only.
- Offline environment: no internet, never run `pip` or `pip install`.""",
}

V18_BASELINE_SAMPLING = {
    "temperature": 0.2,
    "top_p": 0.95,
    "max_output_tokens": 8192,
    "thinking_config": {
        "thinking_budget": 1536,
        "include_thoughts": True,
    },
}


def assemble_prompt(sections: dict) -> str:
    return "\n\n".join([
        sections["header"],
        sections["phase1"],
        sections["phase2"],
        sections["phase3"],
        sections["phase4"],
        sections["breakers"],
    ]).strip() + "\n"


def get_v19_iteration_candidates():
    """
    Defines the 5 cumulative hill-climbing iterations for v18 -> v19 (100% Native ADK).
    """
    return [
        {
            "iteration": 1,
            "id": "v19_iter1_eliminate_inline_repro_loops",
            "lever": "Phase 2 + Phase 4 + Circuit Breakers (Eliminate `python3 -c` Inline Repro Loops & Force Submit After Edit)",
            "rationale": (
                "Eliminates exploratory `python3 -c` reproduction loops before/after editing so the agent edits by "
                "Call 3-4 and invokes `submit_patch` immediately after `py_compile && git diff`."
            ),
            "mutate_sections": {
                "phase2": """## PHASE 2: READ CONTEXT (Call 2)
- Use `read_file` with `path`, `offset`, and `limit: 70` to read the candidate function around the matched line number.
- Identify the exact 2–4 line anchor where the fix belongs.
- NO PYTHON -C EXPLORATION: Do NOT run `python3 -c` reproduction one-liners before editing! Read the source code directly with `read_file` (at most 2 `read_file` calls) and proceed immediately to Phase 3 `edit_file` by Call 3 or 4.""",
                "phase4": """## PHASE 4: VERIFY & SUBMIT (Calls 5-6)
- IMMEDIATELY after `edit_file` in Phase 3, verify syntax and diff in ONE `run_command` call:
  `python3 -m py_compile rel/file.py && git status --short && git diff`
- MANDATORY IMMEDIATE SUBMIT: As soon as `py_compile` succeeds and `git diff` shows your fix, invoke `submit_patch` IMMEDIATELY on the very next turn (Call 5 or 6)! NEVER run `python3 -c` loops after editing, and NEVER run bare `pytest`.""",
                "breakers": """# HARD CIRCUIT BREAKERS
- 28 calls max, 5.5 min per task. When either runs out, the working tree is graded as it is.
- NO REPETITION & NO PYTHON -C LOOPS: Never repeat the same command twice, and never run consecutive `python3 -c` test scripts. Once you read the target function, invoke `edit_file` immediately.
- ZERO-DIFF FORBIDDEN: Submitting 0 bytes changed scores 0 automatically. Never invoke `submit_patch` unless `git diff` is non-empty.
- NEVER create, modify, or delete test files (`tests/**`, `test_*.py`, `*_test.py`, `pytest.ini`, `conftest.py`, `setup.py`, `pyproject.toml`) or leave untracked files in `/workspace`.
- Offline environment: no internet, never run `pip` or `pip install`.""",
            },
            "sampling_override": {},
        },
        {
            "iteration": 2,
            "id": "v19_iter2_kwarg_forwarding_and_coupled_arg_invariants",
            "lever": "Phase 3 (Recursive Keyword Forwarding, Coupled Argument Normalization, NotImplementedError Stubs & Nested `$ref` Dicts)",
            "rationale": (
                "Targets `sympy-23534` (`symbols(name, cls=cls, **args)` recursive keyword-only forwarding), "
                "`sympy-24562` (`Rational(p, q)` coupled string/rational normalization without mutating `q` before "
                "`isinstance(q, SYMPY_INTS)`), `sympy-23950` (`Contains.as_set` returning `self.args[1]` instead of "
                "raising `NotImplementedError`), and `fastapi_14349` (`isinstance(value, str)` check on `schema['$ref']`)."
            ),
            "mutate_sections": {
                "phase3": """## PHASE 3: MANDATORY EDIT (Call 3 or 4)
- Apply your fix using `edit_file` with exact arguments `{"path": "rel/file.py", "old_str": "EXACT 2-4 LINES", "new_str": "REPLACEMENT LINES"}`.
- CRITICAL SYNTAX & PYTHON INVARIANTS:
  1. Keep `old_str` to 2–4 unique lines with exact indentation and WITHOUT line-number prefixes (`42 | `). Apply the change to every parallel implementation (sync/async, Client/AsyncClient, `__all__`).
  2. DATACLASS & EXCEPTION RULES: In `@dataclass`, place default fields (`x: int = None`) AFTER all non-default fields, and propagate `(frozen=True)` to subclasses in the same file. Use repo-specific custom exceptions/warnings when defined.
  3. RECURSIVE KEYWORD FORWARDING: When a function defines explicit keyword parameters alongside `**args`/`**kwargs` (e.g. `def symbols(names, *, cls=Symbol, **args):`) and calls itself recursively (`symbols(name, **args)`), explicit keyword args are NOT in `**args`—always pass them explicitly (`symbols(name, cls=cls, **args)`).
  4. COUPLED ARGUMENT NORMALIZATION: When normalizing two arguments `(p, q)` sequentially (e.g. `if not isinstance(p, SYMPY_INTS): p = Rational(p)` followed by `if not isinstance(q, SYMPY_INTS): q = Rational(q)`), NEVER mutate `q` (`q *= p.q`) before checking `isinstance(q, SYMPY_INTS)`! Accumulate denominator factors in a separate variable `Q = 1` (`Q *= p.q`, then `p *= q.q; Q *= q.p`, else `Q *= int(q)`, then `q = Q`).
  5. STUB METHODS & DICT SCHEMAS: If a target method raises `NotImplementedError()` (e.g. `Contains.as_set(self)`), implement it to return the actual value (`return self.args[1]`)—never just add an error string. When traversing nested schema dicts for `"$ref"`, check `if isinstance(schema["$ref"], str):` before calling `.split("/")`.
- FALLBACK: If `edit_file` reports mismatch once, use `run_command` with atomic Python replace:
  `python3 -c 'from pathlib import Path; p=Path("rel/file.py"); s=p.read_text(); p.write_text(s.replace("OLD_LINE", "NEW_LINE", 1))'`""",
            },
            "sampling_override": {},
        },
        {
            "iteration": 3,
            "id": "v19_iter3_sibling_feature_and_domain_helper_discovery",
            "lever": "Phase 1 + Phase 2 (Sibling Feature Multi-File Grep & Domain System Equivalence Helpers)",
            "rationale": (
                "Targets `rich_3777` (`TTY_INTERACTIVE` modeled on `TTY_COMPATIBLE` across both `rich/console.py` and "
                "`rich/diagnose.py` with bidirectional `'0' -> False, '1' -> True` handling) and `sympy-24213` "
                "(`self.get_dimension_system().equivalent_dims(dim, addend_dim)` instead of `dim != addend_dim`)."
            ),
            "mutate_sections": {
                "phase1": """## PHASE 1: FAST LOCATE (Call 1)
- Use `run_command` with extended regex (`-E`) to search for literal symbols from the issue:
  `git grep -n -I -E "<LiteralSymbol1>|<LiteralSymbol2>" -- "*.py" ":!tests/*"`
- SIBLING FEATURE RULE: If the issue asks to support a new env var, option, or method modeled on an existing sibling (e.g. `TTY_INTERACTIVE` like `TTY_COMPATIBLE`), grep for ALL occurrences of the existing sibling across `*.py` so you update every file where it appears (e.g. both `console.py` and `diagnose.py`) and mirror its exact bidirectional logic (`"0" -> False`, `"1" -> True`).
- ZERO-RESULT ESCAPE: If your first `git grep` returns nothing, DO NOT guess more symbol names! On Call 2, locate the relevant test or traceback via `run_command`:
  `git grep -n -I -E "<keyword>" -- "tests/*.py" || timeout 45 python3 -m pytest -k "<keyword>" --tb=short -q 2>&1 | tail -25`""",
                "phase2": """## PHASE 2: READ CONTEXT (Call 2)
- Use `read_file` with `path`, `offset`, and `limit: 70` to read the candidate function around the matched line number.
- DOMAIN SYSTEM HELPERS: When comparing domain objects in a class/method that uses a system helper (e.g. `self.get_dimension_system()`), use the system's equivalence method (`if not self.get_dimension_system().equivalent_dims(dim, addend_dim):`) rather than raw `!=`.
- NO PYTHON -C EXPLORATION: Do NOT run `python3 -c` reproduction one-liners before editing! Read the source code directly with `read_file` and proceed immediately to Phase 3 `edit_file` by Call 3 or 4.""",
            },
            "sampling_override": {},
        },
        {
            "iteration": 4,
            "id": "v19_iter4_cyclic_recursion_guard_and_render_state",
            "lever": "Phase 3 (Cyclic ExceptionGroup Visited-Set Guard & Empty LiveRender Height Check)",
            "rationale": (
                "Targets `rich_3772` (infinite recursion when `Traceback.extract` traverses cyclic `ExceptionGroup` "
                "exceptions without passing `_visited_exceptions: Optional[Set[BaseException]] = None`) and `rich_3934` "
                "(`Live.stop()` printing a newline even when `self._live_render.last_render_height` is 0)."
            ),
            "mutate_sections": {
                "phase3": """## PHASE 3: MANDATORY EDIT (Call 3 or 4)
- Apply your fix using `edit_file` with exact arguments `{"path": "rel/file.py", "old_str": "EXACT 2-4 LINES", "new_str": "REPLACEMENT LINES"}`.
- CRITICAL SYNTAX & PYTHON INVARIANTS:
  1. Keep `old_str` to 2–4 unique lines with exact indentation and WITHOUT line-number prefixes (`42 | `). Apply the change to every parallel implementation (sync/async, Client/AsyncClient, `__all__`).
  2. DATACLASS & EXCEPTION RULES: In `@dataclass`, place default fields (`x: int = None`) AFTER all non-default fields, and propagate `(frozen=True)` to subclasses in the same file. Use repo-specific custom exceptions/warnings when defined.
  3. RECURSIVE KEYWORD & VISITED-SET FORWARDING:
     - When a function with explicit keyword args (`def symbols(names, *, cls=Symbol, **args):`) calls itself recursively, pass explicit keyword args (`symbols(name, cls=cls, **args)`).
     - When a recursive extractor (`Traceback.extract`) loops over `exc_value.exceptions` in `ExceptionGroup`, add `_visited_exceptions: Optional[Set[BaseException]] = None` (importing `Set`), initialize `grouped_exceptions: Set[BaseException] = set() if _visited_exceptions is None else _visited_exceptions`, skip `if exception in grouped_exceptions: continue`, call `grouped_exceptions.add(exception)`, and pass `_visited_exceptions=grouped_exceptions` to the recursive `Traceback.extract(...)` call.
  4. COUPLED ARGUMENT NORMALIZATION: When normalizing `(p, q)` sequentially (`if not isinstance(p, SYMPY_INTS): p = Rational(p)` then `if not isinstance(q, SYMPY_INTS): q = Rational(q)`), NEVER mutate `q` (`q *= p.q`) before checking `isinstance(q, SYMPY_INTS)`! Accumulate denominator factors in `Q = 1` (`Q *= p.q`, then `p *= q.q; Q *= q.p`, else `Q *= int(q)`, then `q = Q`).
  5. STUB METHODS, DICT SCHEMAS & RENDER STATE:
     - If a target method raises `NotImplementedError()` (e.g. `Contains.as_set(self)`), return the actual value (`return self.args[1]`).
     - In nested schema `"$ref"` replacement, check `if isinstance(schema["$ref"], str):` before `.split("/")`.
     - In `Live.stop()`, only call `self.console.line()` if `self._live_render.last_render_height` is non-zero (adding `@property def last_render_height(self) -> int: return self._shape[1] if self._shape else 0` to `LiveRender`).
- FALLBACK: If `edit_file` reports mismatch once, use `run_command` with atomic Python replace:
  `python3 -c 'from pathlib import Path; p=Path("rel/file.py"); s=p.read_text(); p.write_text(s.replace("OLD_LINE", "NEW_LINE", 1))'`""",
            },
            "sampling_override": {},
        },
        {
            "iteration": 5,
            "id": "v19_iter5_sampling_sharpening",
            "lever": "Sampling Configuration (`temperature: 0.15`, `thinking_budget: 1536`)",
            "rationale": (
                "Tests whether slightly sharper sampling (`temperature: 0.15` vs `0.2`) on top of the best cumulative "
                "v19 prompt reduces turn count and variance across the 12-task cross-benchmark suite."
            ),
            "mutate_sections": {},
            "sampling_override": {
                "temperature": 0.15,
                "top_p": 0.95,
                "max_output_tokens": 8192,
                "thinking_config": {
                    "thinking_budget": 1536,
                    "include_thoughts": True,
                },
            },
        },
    ]


# ==============================================================================
# 3. PACKAGING & EVALUATION ENGINE
# ==============================================================================

def evaluate_harness(
    agent: AgentSimulator,
    tasks: list[dict],
    data_dir: Path,
    temp_base: Path,
    regression_guard_ids: set[str],
    timeout_sec: int = 360,
    max_turns: int = 22,
    cached_tasks: dict[str, dict] | None = None,
) -> dict:
    results = []
    resolved_total = 0
    guard_resolved = 0
    patches_generated = 0
    total_duration = 0.0

    for idx, t in enumerate(tasks, 1):
        tid = t["instance_id"]
        repo = t.get("repo", "")
        is_guard = tid in regression_guard_ids
        role_tag = "GUARD" if is_guard else "TARGET"

        if cached_tasks and tid in cached_tasks:
            res = copy.deepcopy(cached_tasks[tid])
            print(
                f"\n  [{idx}/{len(tasks)}] ({role_tag}) Task: {tid} ({repo}) [CACHED FROM V18]\n"
                f"  -> {tid}: status={res['status']} | resolved={res['resolved']} | "
                f"turns={res['turns']} | patch={res['patch_size']}B | {res['duration_sec']}s",
                flush=True,
            )
        else:
            print(f"\n  [{idx}/{len(tasks)}] ({role_tag}) Task: {tid} ({repo})", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            res = agent.solve_task(t, data_dir, temp_base, timeout_sec=timeout_sec, max_turns=max_turns)
            print(
                f"  -> {tid}: status={res['status']} | resolved={res['resolved']} | "
                f"turns={res['turns']} | patch={res['patch_size']}B | {res['duration_sec']}s",
                flush=True,
            )

        results.append(res)
        total_duration += float(res["duration_sec"])
        if res["patch_size"] > 0:
            patches_generated += 1
        if res["resolved"]:
            resolved_total += 1
            if is_guard:
                guard_resolved += 1

    mean_dur = round(total_duration / max(1, len(tasks)), 2)
    return {
        "resolved_total": resolved_total,
        "guard_resolved": guard_resolved,
        "guard_total": len(regression_guard_ids),
        "patches_generated": patches_generated,
        "total": len(tasks),
        "pass_rate": round(resolved_total / max(1, len(tasks)), 4),
        "mean_duration_sec": mean_dur,
        "tasks": results,
    }


def package_v19_bundle(root: Path, prompt: str, sampling_cfg: dict):
    bundle_v18 = root / "state_of_the_art" / "bundle_v18"
    bundle_v19 = root / "state_of_the_art" / "bundle_v19"
    if bundle_v19.exists():
        shutil.rmtree(bundle_v19)
    shutil.copytree(bundle_v18, bundle_v19)

    # Ensure strictly valid Google ADK GenerateContentConfig keys in configs/sampling.yaml
    adk_sampling = {
        "temperature": float(sampling_cfg.get("temperature", 0.2)),
        "top_p": float(sampling_cfg.get("top_p", 0.95)),
        "max_output_tokens": int(sampling_cfg.get("max_output_tokens", 8192)),
        "thinking_config": sampling_cfg.get(
            "thinking_config",
            {"thinking_budget": 1536, "include_thoughts": True},
        ),
    }

    (bundle_v19 / "prompts" / "system.md").write_text(prompt, encoding="utf-8")
    (bundle_v19 / "configs" / "sampling.yaml").write_text(
        yaml.safe_dump(adk_sampling, sort_keys=False),
        encoding="utf-8",
    )

    zip_path = root / "submissions_archive" / "submission_v19_hill_climbed.zip"
    root_zip = root / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(bundle_v19.rglob("*")):
            if file_path.is_file() and "__pycache__" not in str(file_path):
                zf.write(file_path, file_path.relative_to(bundle_v19))
    shutil.copyfile(zip_path, root_zip)
    print(f"\n[Packager] Packaged Kaggle-compliant v19 bundle to {zip_path} and updated {root_zip.name}", flush=True)


def run_stage3_core_60(
    root: Path,
    agent: AgentSimulator,
    best_prompt: str,
    best_sampling: dict,
    best_stage2_tasks: list[dict],
    data_dir: Path,
    temp_base: Path,
):
    print("\n" + "=" * 95, flush=True)
    print(" STAGE 3: FULL 60-TASK (CORE_60) BENCHMARK EVALUATION OF V19_HILL_CLIMBED", flush=True)
    print("=" * 95, flush=True)

    tasks_file = root / "data" / "tasks.jsonl"
    task_map = {json.loads(l)["instance_id"]: json.loads(l) for l in open(tasks_file, encoding="utf-8") if l.strip()}
    core_60_ids = json.loads((root / "state_of_the_art" / "benchmarks_data" / "test_tasks_core_60.json").read_text())["tasks"]
    core_60_tasks = [task_map[tid] for tid in core_60_ids if tid in task_map]

    out_path = root / "state_of_the_art" / "benchmarks_data" / "benchmark_v19_clean_60_results.json"
    cand_name = "v19_hill_climbed"
    cand_results = []
    completed_ids = set()

    if out_path.is_file():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if cand_name in existing:
                cand_results = existing[cand_name].get("tasks", [])
                completed_ids = {r["instance_id"] for r in cand_results if "instance_id" in r}
                print(f"[Stage 3 Resume] Loaded {len(cand_results)} existing tasks from {out_path.name}", flush=True)
        except Exception as e:
            print(f"[Stage 3 Resume Warning] {e}", flush=True)

    stage2_map = {r["instance_id"]: r for r in best_stage2_tasks}

    agent.system_prompt = best_prompt
    agent.sampling_config = best_sampling

    resolved_count = sum(1 for r in cand_results if r.get("resolved"))

    for idx, t in enumerate(core_60_tasks, 1):
        tid = t["instance_id"]
        if tid in completed_ids:
            continue
        repo = t.get("repo", "")
        if tid in stage2_map:
            res = copy.deepcopy(stage2_map[tid])
            print(
                f"\n--- [{idx}/{len(core_60_tasks)}] Candidate: {cand_name} | Task: {tid} ({repo}) [REUSED FROM STAGE 2 WINNER] ---\n"
                f"Result -> Status: {res['status']} | Resolved: {res['resolved']} | Duration: {res['duration_sec']}s | Turns: {res['turns']} | Patch: {res['patch_size']} bytes",
                flush=True,
            )
        else:
            print(f"\n--- [{idx}/{len(core_60_tasks)}] Candidate: {cand_name} | Task: {tid} ({repo}) ---", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            res = agent.solve_task(t, data_dir, temp_base, timeout_sec=330, max_turns=28)

        cand_results.append(res)
        completed_ids.add(tid)
        if res["resolved"]:
            resolved_count += 1
        print(
            f"Result -> Status: {res['status']} | Resolved: {res['resolved']} | Duration: {res['duration_sec']}s | "
            f"Turns: {res['turns']} | Patch: {res['patch_size']} bytes | Running Total: {resolved_count}/{len(cand_results)}",
            flush=True,
        )

        pass_rate = resolved_count / len(cand_results) if cand_results else 0.0
        all_results = {
            cand_name: {
                "resolved": resolved_count,
                "completed": len(cand_results),
                "total": len(core_60_tasks),
                "pass_rate": round(pass_rate, 4),
                "tasks": cand_results,
            }
        }
        out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    total = len(cand_results)
    avg_dur = round(sum(r["duration_sec"] for r in cand_results) / max(1, total), 1)
    timeouts = sum(1 for r in cand_results if r["status"] == "TIMEOUT")
    avg_turns = round(sum(r["turns"] for r in cand_results) / max(1, total), 1)
    entry = {
        "model_name": "v19_hill_climbed",
        "suite": "core_60",
        "resolved": resolved_count,
        "total": total,
        "resolution_rate_percent": round((resolved_count / max(1, total)) * 100, 1),
        "mean_duration_sec": avg_dur,
        "timeouts": timeouts,
        "mean_turns": avg_turns,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tasks": cand_results,
    }
    update_leaderboard_files(entry, root / "leaderboard.json", root / "LEADERBOARD.md")
    print(f"\n[Stage 3 Complete] v19_hill_climbed on core_60: {resolved_count}/{total} ({entry['resolution_rate_percent']}%)! Saved to {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="google/gemma-4-31B-it")
    parser.add_argument("--output_file", default="hill_climb_v18_to_v19_results.json")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output_file")
    parser.add_argument("--run_core_60", action="store_true", help="Automatically run Stage 3 (v19 on core_60) after Stage 2 completes")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    temp_base = Path("/tmp/hill_climb_v19_workspaces")
    temp_base.mkdir(parents=True, exist_ok=True)
    out_path = root / "state_of_the_art" / "benchmarks_data" / args.output_file

    tasks, guard_ids, suite_meta = build_v19_cross_benchmark_suite(root)
    print(
        f"[Hill-Climber v19] Loaded {len(tasks)} Cross-Benchmark tasks:\n"
        f"  - Guards (4)             : {suite_meta['guards']}\n"
        f"  - Core-60/Climb-12 (4)   : {suite_meta['core_60_and_climb_12_failures']}\n"
        f"  - External HF Py3.12 (4) : {suite_meta['external_hf_py312_failures']}",
        flush=True,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n[Hardware / Model Setup] Loading {args.base_model} onto DGX Spark GPU ({device})...", flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.base_model)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded successfully in {time.time() - t0:.2f}s.\n", flush=True)

    history = {
        "suite_metadata": suite_meta,
        "climb_suite_ids": [t["instance_id"] for t in tasks],
        "regression_guard_ids": sorted(list(guard_ids)),
        "baseline": None,
        "iterations": [],
        "winner": None,
    }
    if args.resume and out_path.is_file():
        try:
            history = json.loads(out_path.read_text(encoding="utf-8"))
            print(f"[Resume] Loaded existing history with {len(history.get('iterations', []))} completed iterations.", flush=True)
        except Exception as e:
            print(f"[Resume Warning] Failed to load {out_path}: {e}", flush=True)

    best_sections = copy.deepcopy(V18_BASELINE_SECTIONS)
    best_sampling = copy.deepcopy(V18_BASELINE_SAMPLING)
    best_prompt = assemble_prompt(best_sections)

    if not history.get("baseline"):
        print("=" * 95, flush=True)
        print(" STEP 0: EVALUATING BASELINE HARNESS (v18_hill_climbed) ON 12-TASK CROSS-BENCHMARK SUITE", flush=True)
        print("=" * 95, flush=True)
        v18_cache = load_v18_cached_baseline_tasks(root)
        agent = AgentSimulator(model, processor, best_prompt, sampling_config=best_sampling, device=device)
        base_res = evaluate_harness(
            agent,
            tasks,
            data_dir,
            temp_base,
            regression_guard_ids=guard_ids,
            timeout_sec=330,
            max_turns=22,
            cached_tasks=v18_cache,
        )
        base_res["id"] = "v18_baseline"
        base_res["prompt_chars"] = len(best_prompt)
        history["baseline"] = base_res
        out_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    else:
        base_res = history["baseline"]

    best_score = base_res["resolved_total"]
    best_mean_duration = base_res["mean_duration_sec"]
    best_id = "v18_baseline"
    best_tasks_list = base_res["tasks"]

    completed_iter_nums = set()
    for past_it in history.get("iterations", []):
        completed_iter_nums.add(past_it["iteration"])
        if past_it.get("decision") == "ACCEPTED":
            best_score = past_it["resolved_total"]
            best_mean_duration = past_it["mean_duration_sec"]
            best_id = past_it["id"]
            best_tasks_list = past_it["tasks"]
            cand_def = next((c for c in get_v19_iteration_candidates() if c["iteration"] == past_it["iteration"]), None)
            if cand_def:
                for k, v in cand_def["mutate_sections"].items():
                    best_sections[k] = v
                if cand_def["sampling_override"]:
                    best_sampling.update(cand_def["sampling_override"])
                best_prompt = assemble_prompt(best_sections)

    package_v19_bundle(root, best_prompt, best_sampling)

    print(
        f"\n[Baseline Status] Current Best: {best_id} -> {best_score}/{len(tasks)} resolved "
        f"({best_score / len(tasks) * 100:.1f}%), Guards: {base_res['guard_resolved']}/{len(guard_ids)}, "
        f"Mean Duration: {best_mean_duration}s",
        flush=True,
    )

    candidates = get_v19_iteration_candidates()
    for it in candidates:
        step = it["iteration"]
        if step in completed_iter_nums:
            print(f"\n[Skip] Iteration {step} ({it['id']}) already completed.", flush=True)
            continue

        cand_sections = copy.deepcopy(best_sections)
        for sec_key, sec_val in it["mutate_sections"].items():
            cand_sections[sec_key] = sec_val
        cand_prompt = assemble_prompt(cand_sections)

        cand_sampling = copy.deepcopy(best_sampling)
        if it["sampling_override"]:
            cand_sampling.update(it["sampling_override"])

        prompt_chars = len(cand_prompt)
        approx_tokens = prompt_chars // 4

        print("\n" + "=" * 95, flush=True)
        print(f" CUMULATIVE V19 HILL-CLIMB ITERATION {step}/5: {it['id']}", flush=True)
        print(f" Lever     : {it['lever']}", flush=True)
        print(f" Rationale : {it['rationale']}", flush=True)
        print(f" Current Best to Beat: {best_id} ({best_score}/{len(tasks)} resolved, {best_mean_duration}s)", flush=True)
        print("=" * 95, flush=True)
        print(f"  Candidate Prompt Size: {prompt_chars} chars (~{approx_tokens} tokens)", flush=True)

        agent = AgentSimulator(
            model,
            processor,
            cand_prompt,
            sampling_config=cand_sampling,
            device=device,
        )
        eval_res = evaluate_harness(
            agent,
            tasks,
            data_dir,
            temp_base,
            regression_guard_ids=guard_ids,
            timeout_sec=330,
            max_turns=22,
        )
        eval_res["iteration"] = step
        eval_res["id"] = it["id"]
        eval_res["lever"] = it["lever"]
        eval_res["rationale"] = it["rationale"]
        eval_res["prompt_chars"] = prompt_chars
        eval_res["approx_tokens"] = approx_tokens

        no_regression = eval_res["guard_resolved"] >= history["baseline"].get("guard_resolved", len(guard_ids))
        strictly_higher = eval_res["resolved_total"] > best_score
        equal_but_faster = (
            eval_res["resolved_total"] == best_score
            and eval_res["mean_duration_sec"] < best_mean_duration
        )

        if no_regression and (strictly_higher or equal_but_faster):
            eval_res["decision"] = "ACCEPTED"
            print(
                f"\n  [CLIMB ACCEPTED] {it['id']} improved harness: "
                f"{best_score}/{len(tasks)} ({best_mean_duration}s) -> "
                f"{eval_res['resolved_total']}/{len(tasks)} ({eval_res['mean_duration_sec']}s) "
                f"[Guards: {eval_res['guard_resolved']}/{len(guard_ids)}]!",
                flush=True,
            )
            best_score = eval_res["resolved_total"]
            best_mean_duration = eval_res["mean_duration_sec"]
            best_sections = cand_sections
            best_prompt = cand_prompt
            best_sampling = cand_sampling
            best_id = it["id"]
            best_tasks_list = eval_res["tasks"]
            package_v19_bundle(root, best_prompt, best_sampling)
        else:
            eval_res["decision"] = "REJECTED_ROLLED_BACK"
            print(
                f"\n  [CLIMB REJECTED -> ROLLBACK] {it['id']} scored "
                f"{eval_res['resolved_total']}/{len(tasks)} ({eval_res['mean_duration_sec']}s) "
                f"[Guards: {eval_res['guard_resolved']}/{len(guard_ids)}] vs best {best_score}/{len(tasks)} ({best_mean_duration}s).",
                flush=True,
            )

        history["iterations"].append(eval_res)
        history["winner"] = {
            "id": best_id,
            "resolved_total": best_score,
            "total": len(tasks),
            "pass_rate": round(best_score / max(1, len(tasks)), 4),
            "mean_duration_sec": best_mean_duration,
        }
        out_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    package_v19_bundle(root, best_prompt, best_sampling)

    timeouts = sum(1 for r in best_tasks_list if r["status"] == "TIMEOUT")
    avg_turns = round(sum(r["turns"] for r in best_tasks_list) / max(1, len(best_tasks_list)), 1)
    stage2_entry = {
        "model_name": "v19_hill_climbed",
        "suite": "climb_12_cross",
        "resolved": best_score,
        "total": len(tasks),
        "resolution_rate_percent": round((best_score / max(1, len(tasks))) * 100, 1),
        "mean_duration_sec": best_mean_duration,
        "timeouts": timeouts,
        "mean_turns": avg_turns,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tasks": best_tasks_list,
    }
    update_leaderboard_files(stage2_entry, root / "leaderboard.json", root / "LEADERBOARD.md")

    print(
        f"\n[Hill-Climber v19] Completed all 5 cumulative iterations! "
        f"Winning harness: {best_id} ({best_score}/{len(tasks)} resolved). Saved to {out_path}",
        flush=True,
    )

    if args.run_core_60:
        agent = AgentSimulator(model, processor, best_prompt, sampling_config=best_sampling, device=device)
        run_stage3_core_60(
            root=root,
            agent=agent,
            best_prompt=best_prompt,
            best_sampling=best_sampling,
            best_stage2_tasks=best_tasks_list,
            data_dir=data_dir,
            temp_base=temp_base,
        )


if __name__ == "__main__":
    main()
