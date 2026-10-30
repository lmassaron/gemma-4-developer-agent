#!/usr/bin/env python3
"""
Local SWE Benchmark & Leaderboard Runner for Gemma 4 Developer Agent Competition.

Features:
- Evaluates candidate models, prompt configs, or zipped submissions across standardized benchmark suites:
  - 'smoke_3': 3 diverse tasks (1 Requests, 1 FastAPI, 1 Rich) for rapid iteration & timeout verification.
  - 'core_12': 12 balanced tasks (4 Requests, 4 FastAPI, 4 Rich) for rigorous local scoring.
  - 'custom': Specific list of instance_ids passed via --tasks.
- Emulates the official competition ADK agent environment:
  - Standard ADK tools: run_command, read_file, edit_file, write_file, get_status, submit_patch.
  - Turn limit (default 20) and strict timeout (default 360s matching the Kaggle evaluation harness).
- Evaluates submitted patches against hidden test suites using pytest.
- Maintains and automatically updates:
  - leaderboard.json (machine-readable historical runs and model comparisons)
  - LEADERBOARD.md (human-readable GitHub-flavored markdown leaderboard table)
"""

import os
import sys
import json
import time
import shutil
import zipfile
import tarfile
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
try:
    from transformers import AutoModelForImageTextToText as AutoModelForMultimodalLM
except ImportError:
    from transformers import AutoModelForMultimodalLM
from transformers import AutoProcessor
from peft import PeftModel


# Standard SWE Benchmark Suites
BENCHMARK_SUITES = {
    "smoke_3": [
        "requests_7205",
        "fastapi_11194",
        "rich_2725",
    ],
    "core_12": [
        "requests_7205",
        "requests_7433",
        "requests_7502",
        "requests_7505",
        "fastapi_11194",
        "fastapi_11355",
        "fastapi_12942",
        "fastapi_13537",
        "rich_2725",
        "rich_2943",
        "rich_3006",
        "rich_3052",
    ],
}


def extract_target_test_info(test_patch: str) -> Tuple[List[str], List[str]]:
    """Extract paths to test files and test function names added in test_patch."""
    files = []
    functions = []
    for line in test_patch.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:].strip()
            if path != "/dev/null":
                files.append(path)
        elif line.startswith("+") and not line.startswith("+++"):
            import re
            m = re.search(r"def\s+(test_[a-zA-Z0-9_]+)", line)
            if m:
                functions.append(m.group(1))
    return list(dict.fromkeys(files)), list(dict.fromkeys(functions))


class TaskWorkspace:
    def __init__(self, task: Dict[str, Any], base_dir: Path, data_dir: Path):
        self.task = task
        self.instance_id = task["instance_id"]
        self.data_dir = data_dir
        self.work_dir = base_dir / self.instance_id
        self.snapshot_tar = data_dir / "snapshots" / f"{self.instance_id}.tgz"
        self.last_read_path = ""

    def setup(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        if self.snapshot_tar.is_file():
            with tarfile.open(self.snapshot_tar, "r:gz") as tar:
                tar.extractall(self.work_dir)
        else:
            repo_name = self.task.get("repo", "")
            base_commit = self.task.get("base_commit", "")
            if not repo_name or not base_commit:
                raise FileNotFoundError(f"Snapshot not found and task lacks repo/base_commit: {self.instance_id}")

            repo_cache_dir = self.data_dir / "repo_cache" / repo_name.replace("/", "_")
            if not repo_cache_dir.is_dir() or not ((repo_cache_dir / "config").is_file() or (repo_cache_dir / ".git").is_dir()):
                repo_cache_dir.parent.mkdir(parents=True, exist_ok=True)
                github_url = f"https://github.com/{repo_name}.git"
                print(f"[RepoCache] Cloning bare {github_url}...", flush=True)
                res = subprocess.run(
                    ["git", "clone", "--bare", github_url, str(repo_cache_dir)],
                    capture_output=True,
                    text=True,
                )
                if res.returncode != 0:
                    raise RuntimeError(f"Failed to clone {github_url}: {res.stderr}")

            # Clone locally with --shared into work_dir and checkout base_commit
            res = subprocess.run(
                ["git", "clone", "--shared", str(repo_cache_dir), str(self.work_dir)],
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                raise RuntimeError(f"Failed to clone shared workspace for {self.instance_id}: {res.stderr}")

            res = subprocess.run(
                ["git", "checkout", "-q", "-f", base_commit],
                cwd=self.work_dir,
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                # Fetch commit if not in default bare clone
                subprocess.run(["git", "fetch", "origin", base_commit], cwd=repo_cache_dir, capture_output=True)
                res = subprocess.run(
                    ["git", "checkout", "-q", "-f", base_commit],
                    cwd=self.work_dir,
                    capture_output=True,
                    text=True,
                )
                if res.returncode != 0:
                    raise RuntimeError(f"Failed to checkout {base_commit} for {self.instance_id}: {res.stderr}")

        # Ensure git configuration
        subprocess.run(["git", "config", "user.name", "evaluator"], cwd=self.work_dir, check=True)
        subprocess.run(["git", "config", "user.email", "eval@kaggle.local"], cwd=self.work_dir, check=True)

    def _normalize_workspace_path(self, path: str) -> str:
        path = (path or "").strip()
        if path.startswith("/workspace/"):
            path = path[len("/workspace/"):]
        elif path == "/workspace":
            path = "."
        return path

    def run_command(self, cmd: str, timeout_sec: int = 30) -> str:
        try:
            # Map /workspace to self.work_dir so Kaggle-native commands run identically locally
            cmd = cmd.replace("/workspace/", f"{self.work_dir}/").replace("/workspace", str(self.work_dir))
            env = os.environ.copy()
            venv_bin = str(Path(sys.executable).parent)
            env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
            python_paths = [str(self.work_dir), str(self.work_dir / "src")]
            if env.get("PYTHONPATH"):
                python_paths.append(env["PYTHONPATH"])
            env["PYTHONPATH"] = ":".join(python_paths)

            res = subprocess.run(
                ["bash", "-c", cmd],
                cwd=self.work_dir,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                env=env,
            )
            out = res.stdout + res.stderr
            if not out.strip():
                return "[Command executed successfully with no output]"
            if len(out) > 6000:
                out = out[:2500] + f"\n\n... [TRUNCATED {len(out) - 5000} CHARS] ...\n\n" + out[-2500:]
            return out
        except subprocess.TimeoutExpired:
            return f"[Error: Command timed out after {timeout_sec}s]"
        except Exception as e:
            return f"[Error: {e}]"

    def read_file(self, path: str, offset: int = 0, limit: int = 80) -> str:
        path = self._normalize_workspace_path(path)
        limit = min(int(limit), 100)
        fp = self.work_dir / path
        if not fp.is_file():
            return f"[Error: File not found: {path}]"
        self.last_read_path = path
        try:
            lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
            total = len(lines)
            selected = lines[offset: offset + limit]
            numbered = [f"{offset + idx + 1:4d} | {line}" for idx, line in enumerate(selected)]
            header = f"[File: {path} (lines {offset+1}-{min(total, offset+limit)} of {total})]\n"
            return header + "\n".join(numbered)
        except Exception as e:
            return f"[Error reading file: {e}]"

    def edit_file(self, path: str, old_content: str, new_content: str) -> str:
        path = self._normalize_workspace_path(path)
        fp = self.work_dir / path
        if not fp.is_file():
            return f"[Error: File not found: {path}]"
        try:
            text = fp.read_text(encoding="utf-8")
            if not old_content:
                return f"[Error: old_content is empty]"

            # Tier 1: Exact literal match
            if old_content in text:
                occurrences = text.count(old_content)
                if occurrences > 1:
                    return f"[Error: old_content occurs {occurrences} times in {path}. Please provide a larger unique snippet.]"
                updated = text.replace(old_content, new_content, 1)
                fp.write_text(updated, encoding="utf-8")
                return f"[Successfully updated {path}]"

            # Tier 1b: Strip line numbers if copied from read_file output (e.g., '  42 | def foo')
            import re
            cleaned_old = re.sub(r"^\s*\d+\s*\|\s?", "", old_content, flags=re.MULTILINE)
            cleaned_new = re.sub(r"^\s*\d+\s*\|\s?", "", new_content, flags=re.MULTILINE)
            if (cleaned_old != old_content) and (cleaned_old in text):
                occurrences = text.count(cleaned_old)
                if occurrences > 1:
                    return f"[Error: old_content (line-numbers stripped) occurs {occurrences} times in {path}. Provide larger snippet.]"
                updated = text.replace(cleaned_old, cleaned_new, 1)
                fp.write_text(updated, encoding="utf-8")
                return f"[Successfully updated {path}]"

            # Tier 2: Whitespace and Indentation Normalized Line-Matching (Aider style)
            file_lines = text.splitlines(keepends=True)
            target_lines = [l.strip() for l in cleaned_old.splitlines() if l.strip()]
            if not target_lines:
                return f"[Error: old_content contains only whitespace]"

            matches = []
            for i in range(len(file_lines) - len(target_lines) + 1):
                window = [file_lines[i + j].strip() for j in range(len(target_lines))]
                if window == target_lines:
                    matches.append(i)

            if len(matches) == 1:
                start_idx = matches[0]
                end_idx = start_idx + len(target_lines)
                first_file_line = file_lines[start_idx]
                file_indent = first_file_line[:len(first_file_line) - len(first_file_line.lstrip())]
                first_old_nonempty = next((l for l in cleaned_old.splitlines() if l.strip()), "")
                old_indent = first_old_nonempty[:len(first_old_nonempty) - len(first_old_nonempty.lstrip())]

                new_lines = cleaned_new.splitlines(keepends=True)
                adjusted_new_lines = []
                for nl in new_lines:
                    if not nl.strip():
                        adjusted_new_lines.append(nl if nl.endswith("\n") else nl + "\n")
                    else:
                        if nl.startswith(old_indent):
                            adjusted_new_lines.append(file_indent + nl[len(old_indent):])
                        else:
                            adjusted_new_lines.append(file_indent + nl.lstrip())
                if adjusted_new_lines and not adjusted_new_lines[-1].endswith("\n"):
                    adjusted_new_lines[-1] += "\n"

                updated = "".join(file_lines[:start_idx] + adjusted_new_lines + file_lines[end_idx:])
                fp.write_text(updated, encoding="utf-8")
                return f"[Successfully updated {path} (via normalized line match)]"
            elif len(matches) > 1:
                return f"[Error: old_content matches {len(matches)} locations in {path}. Please provide a larger unique snippet.]"

            # Check if old_content exists in another python file in workspace
            first_line = target_lines[0] if target_lines else ""
            if first_line and len(first_line) > 8:
                try:
                    grep_res = subprocess.run(
                        ["git", "grep", "-l", "-F", first_line, "--", "*.py", ":!tests/*"],
                        cwd=self.work_dir,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    other_matches = [f.strip() for f in grep_res.stdout.splitlines() if f.strip() and f.strip() != path]
                    if len(other_matches) == 1:
                        target_other = other_matches[0]
                        return f"[Error: old_content not found in {path}, but exact matching code was found in {target_other}! Please call edit_file with path='{target_other}'.]"
                except Exception:
                    pass

            return f"[Error: old_content not found in {path}. Make sure whitespace and indentation match exactly.]"
        except Exception as e:
            return f"[Error editing file: {e}]"

    def write_file(self, path: str, content: str) -> str:
        fp = self.work_dir / path
        fp.parent.mkdir(parents=True, exist_ok=True)
        try:
            fp.write_text(content, encoding="utf-8")
            return f"[Successfully wrote {len(content)} bytes to {path}]"
        except Exception as e:
            return f"[Error writing file: {e}]"

    def get_status(self) -> str:
        try:
            res = subprocess.run(
                ["git", "status", "--short"],
                cwd=self.work_dir,
                capture_output=True,
                text=True,
            )
            diff_stat = subprocess.run(
                ["git", "diff", "--stat"],
                cwd=self.work_dir,
                capture_output=True,
                text=True,
            )
            return f"Git Status:\n{res.stdout}\nDiff Stat:\n{diff_stat.stdout}"
        except Exception as e:
            return f"[Error getting status: {e}]"

    def get_diff(self) -> str:
        try:
            res = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=self.work_dir,
                capture_output=True,
                text=True,
            )
            return res.stdout
        except Exception:
            return ""

    def run_verification(self, timeout_sec: int = 45) -> Tuple[bool, str]:
        """Applies hidden test_patch and runs pytest against target functions."""
        test_patch = self.task.get("test_patch", "")
        test_files, test_functions = extract_target_test_info(test_patch)

        # Apply hidden test patch
        self.work_dir.mkdir(parents=True, exist_ok=True)
        patch_file = self.work_dir / ".hidden_test.patch"
        clean_patch = test_patch if test_patch.endswith("\n") else (test_patch + "\n")
        patch_file.write_text(clean_patch, encoding="utf-8")
        try:
            res = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", "--recount", str(patch_file)],
                cwd=self.work_dir,
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                res = subprocess.run(
                    ["git", "apply", "--whitespace=fix", str(patch_file)],
                    cwd=self.work_dir,
                    capture_output=True,
                    text=True,
                )
            if res.returncode != 0:
                return False, f"Hidden test patch failed to apply: {res.stderr}"
        finally:
            patch_file.unlink(missing_ok=True)

        # Run targeted pytest
        pytest_cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short", "-W", "ignore"]
        if test_files:
            pytest_cmd.extend(test_files)
        if test_functions:
            pytest_cmd.extend(["-k", " or ".join(test_functions)])

        env = os.environ.copy()
        python_paths = [str(self.work_dir), str(self.work_dir / "src")]
        if env.get("PYTHONPATH"):
            python_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = ":".join(python_paths)

        try:
            proc = subprocess.run(
                pytest_cmd,
                cwd=self.work_dir,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                env=env,
            )
            out = proc.stdout + "\n" + proc.stderr
            return (proc.returncode == 0, out)
        except subprocess.TimeoutExpired:
            return False, f"Pytest timed out after {timeout_sec}s"
        except Exception as e:
            return False, f"Pytest execution error: {e}"

    def cleanup(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)


ADK_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a bash shell command in /workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute in /workspace"}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a slice of lines from a file in /workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file in /workspace"},
                    "offset": {"type": "integer", "description": "0-based starting line offset"},
                    "limit": {"type": "integer", "description": "Maximum number of lines to read"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace old_str with new_str in a file in /workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file in /workspace"},
                    "old_str": {"type": "string", "description": "Exact existing snippet to replace"},
                    "new_str": {"type": "string", "description": "Replacement snippet"},
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file in /workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file in /workspace"},
                    "content": {"type": "string", "description": "Full file content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_status",
            "description": "Run git status --short and git diff --stat in /workspace.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_patch",
            "description": "Submit the current git diff patch and finalize the task.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


class AgentSimulator:
    """Multi-turn Gemma 4 Agent Rollout Simulator."""

    def __init__(self, model, processor, system_prompt: str, sampling_config: Optional[Dict[str, Any]] = None, device: str = "cuda:0"):
        self.model = model
        self.processor = processor
        self.system_prompt = system_prompt
        self.sampling_config = sampling_config or {}
        self.device = device

    def parse_tool_call(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Parses Gemma 4 native <|tool_call>call:name{key:<|"|>val<|"|>}<tool_call|> syntax or JSON/Python fallback."""
        import ast
        import re
        import yaml

        # Strip internal thought channel if present (<|channel>thought ... <channel|>)
        if "<channel|>" in text:
            cleaned_text = text.split("<channel|>")[-1].strip()
        else:
            cleaned_text = re.sub(r"<\|channel\>thought[\s\S]*?(?:<channel>|<channel\|>|<\|channel\|>|$)", "", text).strip()
        search_text = cleaned_text if cleaned_text else text

        # Pattern 0: Canonical Gemma 4 Native Tool Call: <|tool_call>call:tool_name{key:<|"|>val<|"|>}<tool_call|>
        m_native = re.search(r"(?:<\|tool_call\|?>)?call:(run_command|read_file|edit_file|write_file|get_status|submit_patch)\{([\s\S]*?)\}(?:<tool_call\|?>|<\|tool_call\|?>)", search_text)
        if not m_native and "<|\"|>" in search_text:
            m_native = re.search(r"(?:call:)?(run_command|read_file|edit_file|write_file|get_status|submit_patch)\{([\s\S]*?)\}", search_text)
        if m_native and ("<|\"|>" in m_native.group(2) or not m_native.group(2).strip() or not m_native.group(2).strip().startswith('"')):
            tool_name = m_native.group(1)
            body = m_native.group(2).strip()
            if not body:
                return tool_name, {}
            args = {}
            pos = 0
            n = len(body)
            parsed_any = False
            while pos < n:
                while pos < n and body[pos] in " ,\n\r\t":
                    pos += 1
                if pos >= n:
                    break
                km = re.match(r"([a-zA-Z0-9_]+)\s*:\s*", body[pos:])
                if not km:
                    break
                key = km.group(1)
                pos += km.end()
                if body[pos:].startswith("<|\"|>"):
                    pos += 5
                    end_idx = body.find("<|\"|>", pos)
                    if end_idx == -1:
                        val = body[pos:]
                        pos = n
                    else:
                        val = body[pos:end_idx]
                        pos = end_idx + 5
                    args[key] = val
                    parsed_any = True
                else:
                    vm = re.match(r"([^,}]+)", body[pos:])
                    if vm:
                        raw_val = vm.group(1).strip()
                        pos += vm.end()
                        if raw_val.isdigit() or (raw_val.startswith("-") and raw_val[1:].isdigit()):
                            args[key] = int(raw_val)
                        elif raw_val.lower() == "true":
                            args[key] = True
                        elif raw_val.lower() == "false":
                            args[key] = False
                        else:
                            try:
                                args[key] = float(raw_val)
                            except Exception:
                                args[key] = raw_val.strip("\"'")
                        parsed_any = True
                    else:
                        break
            if parsed_any:
                return tool_name, args

        # Pattern 1: call:default_api:tool_name{args}, call:tool_name{args}, or bare tool_name{args}
        m = re.search(r"(?:call:(?:default_api:)?)?\b(run_command|read_file|edit_file|write_file|get_status|submit_patch)\s*(\{[\s\S]*\})", search_text)
        if m:
            tool_name = m.group(1)
            raw_args = m.group(2)
            # Also try non-greedy first if greedy fails, or vice versa
            m_non_greedy = re.search(r"(?:call:(?:default_api:)?)?\b(run_command|read_file|edit_file|write_file|get_status|submit_patch)\s*(\{[\s\S]*?\})", search_text)
            candidates_raw = [raw_args]
            if m_non_greedy and m_non_greedy.group(2) != raw_args:
                candidates_raw.insert(0, m_non_greedy.group(2))
            for cand_raw in candidates_raw:
                try:
                    args = json.loads(cand_raw)
                    if isinstance(args, dict):
                        return tool_name, args
                except Exception:
                    pass
                try:
                    args = yaml.safe_load(cand_raw)
                    if isinstance(args, dict):
                        return tool_name, args
                except Exception:
                    pass
            m_cmd = re.search(r'command\s*:\s*["\']([\s\S]+?)["\']\s*\}?$', raw_args)
            if m_cmd:
                return tool_name, {"command": m_cmd.group(1)}
            return tool_name, {"command": raw_args.strip("{} \n\t")}

        # Pattern 1b: Python function call syntax: tool_name(key="val", key2=123)
        m_py = re.search(r"\b(read_file|edit_file|write_file|run_command|get_status|submit_patch)\s*\(([\s\S]*?)\)", search_text)
        if m_py:
            t_name = m_py.group(1)
            raw_py_args = m_py.group(2).strip()
            if not raw_py_args:
                return t_name, {}
            try:
                tree = ast.parse(f"f({raw_py_args})")
                call = tree.body[0].value
                py_args = {}
                for kw in call.keywords:
                    py_args[kw.arg] = ast.literal_eval(kw.value)
                return t_name, py_args
            except Exception:
                pass

        # Pattern 1c: Direct shell command call: call:<shell_cmd> (standard Kaggle shell evaluation)
        m_raw = re.search(r"(?:<\|tool_call\>)?call:(?!submit_patch\b)([^<{\n\r][^<\n]*?)(?:<tool_call\|>|$)", search_text)
        if m_raw:
            raw_cmd = m_raw.group(1).strip()
            if raw_cmd and not raw_cmd.startswith("{") and not raw_cmd.startswith("("):
                return "run_command", {"command": raw_cmd}

        # Pattern 2: JSON-style tool call block: {"name": "tool_name", "arguments": {...}}
        m_json = re.search(r"\{[^{}]*\"name\"\s*:\s*\"([^\"]+)\"[^{}]*\"arguments\"\s*:\s*(\{[\s\S]*?\})[^{}]*\}", search_text)
        if m_json:
            tool_name = m_json.group(1).replace("default_api:", "")
            try:
                args = json.loads(m_json.group(2))
                return tool_name, args
            except Exception:
                pass

        # Pattern 3: explicit submit_patch invocation only (call:submit_patch, submit_patch(), or submit_patch{})
        if re.search(r"\b(?:call:(?:default_api:)?)?submit_patch\s*(?:\(\s*\)|\{\s*\})", search_text):
            return "submit_patch", {}

        return None

    def execute_tool(self, ws: TaskWorkspace, tool_name: str, args: Dict[str, Any]) -> Tuple[str, bool]:
        """Executes a tool on the workspace. Returns (output_string, is_terminal_action)."""
        if tool_name == "run_command":
            cmd = args.get("command") or args.get("cmd") or ""
            cmd = cmd.strip()
            # Clean up accidental 'command:' or '"command":' wrapper
            if cmd.startswith('"command":') or cmd.startswith("'command':"):
                cmd = cmd[10:].strip().strip('"\'')
            elif cmd.startswith("command:"):
                cmd = cmd[8:].strip().strip('"\'')
            if cmd.strip().strip('"\'') in ("submit_patch", "submit_patch()", "command: submit_patch"):
                diff = ws.get_diff()
                if not diff.strip():
                    return "[Error: Cannot submit an empty patch! git diff shows 0 bytes changed. Your edit did not apply. You MUST edit source files before submitting.]", False
                return "[Patch submitted. Finalizing task evaluation.]", True
            if cmd.strip() in ("pytest", "python3 -m pytest", "python -m pytest", "pytest tests/", "python3 -m pytest tests/"):
                return "[Error: Running the entire pytest suite without a specific test file or `-k <keyword>` filter is forbidden because it times out and exhausts memory. Pass a specific test file or `-k '<keyword>'`.]", False
            return ws.run_command(cmd), False
        elif tool_name == "read_file":
            path = args.get("path") or args.get("file_path") or args.get("file") or ""
            offset = int(args.get("offset", 0))
            limit = int(args.get("limit", 80))
            return ws.read_file(path, offset, limit), False
        elif tool_name == "edit_file":
            path = args.get("path") or args.get("file_path") or args.get("file") or ""
            if not path and getattr(ws, "last_read_path", ""):
                path = ws.last_read_path
            old_str = args.get("old_str") or args.get("old_content") or args.get("old_string") or ""
            new_str = args.get("new_str") or args.get("new_content") or args.get("new_string") or ""
            return ws.edit_file(path, old_str, new_str), False
        elif tool_name == "write_file":
            path = args.get("path") or args.get("file_path") or args.get("file") or ""
            content = args.get("content") or ""
            return ws.write_file(path, content), False
        elif tool_name == "get_status":
            return ws.get_status(), False
        elif tool_name == "submit_patch":
            diff = ws.get_diff()
            if not diff.strip():
                return "[Error: Cannot submit an empty patch! git diff shows 0 bytes changed. Your edit did not apply. You MUST edit source files before submitting.]", False
            return "[Patch submitted. Finalizing task evaluation.]", True
        else:
            return f"[Unknown tool: {tool_name}]", False

    def solve_task(self, task: Dict[str, Any], data_dir: Path, temp_dir: Path, timeout_sec: int = 360, max_turns: int = 20) -> Dict[str, Any]:
        """Runs the complete agent loop on a single task."""
        ws = TaskWorkspace(task, temp_dir, data_dir)
        ws.setup()
        start_time = time.time()
        tool_calls_log = []

        issue_text = task.get("problem_statement") or task.get("issue") or ""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"# PROBLEM STATEMENT\n{issue_text}"},
        ]

        turns_used = 0
        terminal_invoked = False

        try:
            while turns_used < max_turns:
                elapsed = time.time() - start_time
                if elapsed > timeout_sec:
                    return {
                        "instance_id": task["instance_id"],
                        "repo": task["repo"],
                        "status": "TIMEOUT",
                        "resolved": False,
                        "duration_sec": round(elapsed, 2),
                        "turns": turns_used,
                        "tool_calls": tool_calls_log,
                        "patch_size": len(ws.get_diff()),
                    }

                turns_used += 1

                # Generate model turn using Kaggle-faithful Gemma 4 native tool declarations
                prompt_str = self.processor.apply_chat_template(
                    messages,
                    tools=ADK_TOOL_SCHEMAS,
                    enable_thinking=False,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if prompt_str.endswith("<tool_response|>"):
                    prompt_str += "<|channel>thought\n<channel|>"
                inputs = self.processor.tokenizer(prompt_str, return_tensors="pt").to(self.device)

                temp = float(self.sampling_config.get("temperature", 0.2))
                top_p = float(self.sampling_config.get("top_p", 0.95))
                do_sample = temp > 0.0
                max_toks = int(self.sampling_config.get("max_new_tokens") or self.sampling_config.get("max_output_tokens") or 1024)

                gen_kwargs = {
                    "input_ids": inputs.input_ids,
                    "attention_mask": inputs.attention_mask,
                    "max_new_tokens": min(max_toks, 1024),
                    "do_sample": do_sample,
                    "repetition_penalty": float(self.sampling_config.get("repetition_penalty", 1.05)),
                    "eos_token_id": [1, 106, 49, 50],
                }
                if do_sample:
                    gen_kwargs["temperature"] = temp
                    gen_kwargs["top_p"] = top_p
                with torch.no_grad():
                    try:
                        gen_ids = self.model.generate(**gen_kwargs)
                    except torch.OutOfMemoryError:
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                        gen_kwargs["max_new_tokens"] = 512
                        gen_ids = self.model.generate(**gen_kwargs)
                    except Exception:
                        gen_kwargs["mm_token_type_ids"] = torch.zeros_like(inputs.input_ids)
                        gen_ids = self.model.generate(**gen_kwargs)

                response_text = self.processor.tokenizer.decode(
                    gen_ids[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=False,
                )
                del inputs, gen_ids
                torch.cuda.empty_cache()

                parsed = self.parse_tool_call(response_text)
                if not parsed:
                    preview = response_text.replace("\n", " ").strip()[:140]
                    print(f"    Turn {turns_used:02d} | [NO TOOL PARSED] Preview: {preview}", flush=True)
                    tool_calls_log.append(f"[NO_TOOL]: {preview}")
                    messages.append({"role": "model", "content": response_text})
                    messages.append({"role": "user", "content": "Please invoke a tool: call:tool_name{arguments}."})
                    continue

                tool_name, tool_args = parsed
                tool_calls_log.append(f"{tool_name}: {tool_args}")
                print(f"    Turn {turns_used:02d} | Call: {tool_name} -> {tool_args}", flush=True)

                # Execute tool
                tool_output, is_terminal = self.execute_tool(ws, tool_name, tool_args)

                # Anti-repetition loop breaker: prevent repeating ANY identical action consecutively or repeating failed searches
                current_sig = f"{tool_name}: {tool_args}"
                is_repeated_consecutive = (len(tool_calls_log) >= 2 and tool_calls_log[-2] == current_sig)
                is_empty_repeated_cmd = False
                if tool_name == "run_command" and (not tool_output.strip() or tool_output.strip() == "[Command executed successfully with no output]"):
                    cmd_str = tool_args.get("command", "")
                    count_same_cmd = sum(1 for call in tool_calls_log if f"'command': '{cmd_str}'" in call or f'"command": "{cmd_str}"' in call)
                    if count_same_cmd >= 2:
                        is_empty_repeated_cmd = True

                if is_repeated_consecutive or is_empty_repeated_cmd:
                    if tool_name == "submit_patch":
                        tool_output = "[CIRCUIT BREAKER]: submit_patch was repeatedly rejected because git diff is 0 bytes! Calling submit_patch again will NEVER work. You MUST invoke edit_file now to apply your source fix."
                    elif tool_name == "read_file":
                        tool_output = f"[CIRCUIT BREAKER]: You already read '{tool_args.get('path')}'. DO NOT read it again. You MUST apply an edit via edit_file now."
                    elif tool_name == "run_command":
                        cmd_str = tool_args.get("command", "")
                        if "python" in cmd_str and "pytest" not in cmd_str and "py_compile" not in cmd_str:
                            tool_output = "[CIRCUIT BREAKER]: Repeated python3 -c command blocked! Stop running exploratory python3 -c one-liners. Call edit_file NOW to apply your fix, or call submit_patch if you already edited the source file."
                        else:
                            tool_output = "[CIRCUIT BREAKER]: Command returned 0 results or was repeated identically. Do NOT repeat the same command. Read the target file with read_file, apply your fix with edit_file, or call submit_patch."
                    else:
                        tool_output = f"[CIRCUIT BREAKER]: Repeated identical tool call '{tool_name}' detected. You MUST change your tool action."

                if "[Error" in tool_output or "[CIRCUIT BREAKER]" in tool_output:
                    err_snip = tool_output.strip().splitlines()[0][:140]
                    print(f"         └── Tool Error: {err_snip}", flush=True)

                tc_id = f"tc_{turns_used}"
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": tc_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": tool_args},
                        }
                    ],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": tool_name,
                    "content": tool_output,
                })

                if is_terminal:
                    terminal_invoked = True
                    break

            # Run test verification
            elapsed = time.time() - start_time
            source_patch = ws.get_diff()
            passed, test_output = ws.run_verification()
            patch_text = source_patch
            if not passed and patch_text.strip():
                fail_snip = test_output.strip().splitlines()[-1][:140] if test_output else ""
                print(f"         └── Test Verification Failed: {fail_snip}", flush=True)

            has_patch = bool(patch_text.strip())
            resolved = passed and has_patch
            status = "RESOLVED" if resolved else ("FAILED" if has_patch else "NO_PATCH")

            return {
                "instance_id": task["instance_id"],
                "repo": task["repo"],
                "status": status,
                "resolved": resolved,
                "duration_sec": round(elapsed, 2),
                "turns": turns_used,
                "tool_calls": tool_calls_log,
                "patch_size": len(patch_text),
                "terminal_invoked": terminal_invoked,
                "output_snippet": test_output[-400:] if test_output else "",
            }

        finally:
            ws.cleanup()
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def update_leaderboard_files(entry: Dict[str, Any], json_path: Path, md_path: Path):
    """Updates leaderboard.json and LEADERBOARD.md with new benchmark results."""
    data = {"leaderboard": [], "runs": {}}
    if json_path.is_file():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            data = {"leaderboard": [], "runs": {}}

    # Update summary table
    runs = data.setdefault("runs", {})
    run_key = f"{entry['model_name']}_{entry['suite']}_{entry['timestamp']}"
    runs[run_key] = entry

    # Upsert in leaderboard list (keep highest resolution rate per model+suite)
    existing_idx = None
    for idx, item in enumerate(data["leaderboard"]):
        if item["model_name"] == entry["model_name"] and item["suite"] == entry["suite"]:
            existing_idx = idx
            break

    summary_item = {
        "model_name": entry["model_name"],
        "suite": entry["suite"],
        "resolved": entry["resolved"],
        "total": entry["total"],
        "resolution_rate_percent": entry["resolution_rate_percent"],
        "mean_duration_sec": entry["mean_duration_sec"],
        "timeouts": entry["timeouts"],
        "mean_turns": entry["mean_turns"],
        "timestamp": entry["timestamp"],
    }

    if existing_idx is not None:
        data["leaderboard"][existing_idx] = summary_item
    else:
        data["leaderboard"].append(summary_item)

    # Sort leaderboard by resolution rate desc, then mean duration asc
    data["leaderboard"].sort(key=lambda x: (-x["resolution_rate_percent"], x["mean_duration_sec"]))
    json_path.write_text(json.dumps(data, indent=2))

    # Generate LEADERBOARD.md
    lines = [
        "# Gemma 4 Developer Agent — Local Benchmark Leaderboard",
        "",
        "> Standardized evaluation of Gemma 4 developer agent candidates on verified SWE-bench tasks under official 360s/task competition constraints.",
        "",
        "## Overall Leaderboard",
        "",
        "| Rank | Model / Configuration | Suite | Resolution Rate | Resolved | Avg Time / Task | Timeouts (>360s) | Avg Turns | Last Evaluated |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for rank, row in enumerate(data["leaderboard"], 1):
        rate_str = f"**{row['resolution_rate_percent']:.1f}%**"
        res_str = f"{row['resolved']}/{row['total']}"
        time_str = f"{row['mean_duration_sec']:.1f}s"
        lines.append(
            f"| #{rank} | **{row['model_name']}** | `{row['suite']}` | {rate_str} | {res_str} | {time_str} | {row['timeouts']} | {row['mean_turns']:.1f} | {row['timestamp']} |"
        )

    lines.extend([
        "",
        "## Benchmark Suites",
        "- **`smoke_3`**: 3 representative tasks across core repositories (`psf/requests`, `fastapi/fastapi`, `Textualize/rich`) for rapid timeout verification and regression testing.",
        "- **`core_12`**: 12 balanced tasks (4 Requests, 4 FastAPI, 4 Rich) providing a high-confidence local signal.",
        "",
        "## Latest Evaluation Run Details",
        f"**Run ID**: `{run_key}`  ",
        f"**Model**: `{entry['model_name']}` | **Suite**: `{entry['suite']}`  ",
        "",
        "| Instance ID | Repository | Status | Resolved | Duration (s) | Turns | Tool Calls | Patch Size |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for task_res in entry.get("tasks", []):
        res_badge = "✅ PASS" if task_res.get("resolved") else "❌ FAIL"
        lines.append(
            f"| `{task_res['instance_id']}` | `{task_res['repo']}` | `{task_res['status']}` | {res_badge} | {task_res['duration_sec']}s | {task_res['turns']} | {len(task_res.get('tool_calls', []))} | {task_res['patch_size']} B |"
        )

    lines.append("")
    md_path.write_text("\n".join(lines))
    print(f"\n[Leaderboard] Successfully updated {md_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Gemma 4 Local Benchmark & Leaderboard Runner")
    parser.add_argument("--suite", choices=["smoke_3", "core_12", "custom"], default="smoke_3")
    parser.add_argument("--tasks_file", default="data/tasks.jsonl", help="Path to tasks.jsonl")
    parser.add_argument("--data_dir", default="data", help="Root data directory containing snapshots/")
    parser.add_argument("--submission_zip", help="Path to submission.zip to benchmark")
    parser.add_argument("--base_model", default="google/gemma-4-31B-it", help="Base model identifier or path")
    parser.add_argument("--adapter_dir", help="Path to PEFT LoRA adapter directory")
    parser.add_argument("--system_prompt_path", default="submission/prompts/system.md")
    parser.add_argument("--model_name", help="Display name for leaderboard")
    parser.add_argument("--timeout_sec", type=int, default=360, help="Per-task timeout in seconds")
    parser.add_argument("--max_turns", type=int, default=20, help="Max agent turns per task")
    parser.add_argument("--output_file", help="Path to save detailed benchmark JSON results")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    temp_dir = Path("/tmp/gemma4_benchmark_workspaces")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Unpack submission_zip if provided
    system_prompt = ""
    sampling_config = {}
    if args.submission_zip and Path(args.submission_zip).is_file():
        zip_p = Path(args.submission_zip).resolve()
        print(f"[Submission] Reading submission package: {zip_p.name}...")
        with zipfile.ZipFile(zip_p) as z:
            namelist = z.namelist()
            if "prompts/system.md" in namelist:
                system_prompt = z.read("prompts/system.md").decode("utf-8")
                print("  - Loaded system prompt from prompts/system.md")
            if "eval_config.yaml" in namelist:
                import yaml
                ev_data = yaml.safe_load(z.read("eval_config.yaml").decode("utf-8")) or {}
                ev = ev_data.get("evaluation", {})
                if "max_tool_calls" in ev:
                    args.max_turns = int(ev["max_tool_calls"])
                if "max_time_minutes" in ev:
                    args.timeout_sec = int(float(ev["max_time_minutes"]) * 60)
                print(f"  - Configured evaluation limits: max_turns={args.max_turns}, timeout_sec={args.timeout_sec}")
            if "configs/sampling.yaml" in namelist:
                import yaml
                sampling_config = yaml.safe_load(z.read("configs/sampling.yaml").decode("utf-8")) or {}
                print(f"  - Loaded sampling config: {sampling_config}")
        if not args.model_name:
            args.model_name = zip_p.stem

    if not system_prompt:
        system_prompt = Path(args.system_prompt_path).read_text(encoding="utf-8")

    # 2. Load Tasks
    target_ids = BENCHMARK_SUITES.get(args.suite, [])
    task_map = {}
    with open(args.tasks_file, "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            task_map[t["instance_id"]] = t

    if target_ids:
        selected_tasks = [task_map[tid] for tid in target_ids if tid in task_map]
    else:
        selected_tasks = list(task_map.values())
    print(f"[Benchmark] Selected {len(selected_tasks)} tasks for suite '{args.suite}'.")

    # 3. Setup Model & Processor
    model_name = args.model_name or (Path(args.adapter_dir).name if args.adapter_dir else Path(args.base_model).name)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[Model] Loading Processor: {args.base_model}...")
    processor = AutoProcessor.from_pretrained(args.base_model)

    print(f"[Model] Loading Base Model: {args.base_model} on {device}...")
    model = AutoModelForMultimodalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    if args.adapter_dir and Path(args.adapter_dir).is_dir():
        print(f"[Model] Attaching LoRA Adapter from: {args.adapter_dir}...")
        model = PeftModel.from_pretrained(model, args.adapter_dir)

    model.eval()

    # 4. Execute Suite
    agent = AgentSimulator(model, processor, system_prompt, sampling_config=sampling_config, device=device)
    task_results = []
    print("\n" + "=" * 65)
    print(f"      STARTING BENCHMARK SUITE: {args.suite} ({model_name})")
    print("=" * 65)

    for idx, t in enumerate(selected_tasks, 1):
        print(f"\n[{idx}/{len(selected_tasks)}] Evaluating: {t['instance_id']} ({t['repo']})...")
        res = agent.solve_task(t, data_dir, temp_dir, timeout_sec=args.timeout_sec, max_turns=args.max_turns)
        task_results.append(res)
        print(f"--> Status: {res['status']} | Resolved: {res['resolved']} | Duration: {res['duration_sec']}s | Turns: {res['turns']}")

    # 4. Compute Metrics
    resolved_count = sum(1 for r in task_results if r["resolved"])
    total_count = len(task_results)
    rate = (resolved_count / total_count * 100.0) if total_count > 0 else 0.0
    mean_dur = sum(r["duration_sec"] for r in task_results) / total_count if total_count > 0 else 0.0
    timeouts = sum(1 for r in task_results if r["status"] == "TIMEOUT")
    mean_turns = sum(r["turns"] for r in task_results) / total_count if total_count > 0 else 0.0

    entry = {
        "model_name": model_name,
        "suite": args.suite,
        "resolved": resolved_count,
        "total": total_count,
        "resolution_rate_percent": round(rate, 2),
        "mean_duration_sec": round(mean_dur, 2),
        "timeouts": timeouts,
        "mean_turns": round(mean_turns, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tasks": task_results,
    }

    # 5. Update Leaderboard
    if args.output_file:
        out_p = Path(args.output_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2)
        print(f"[Benchmark] Saved detailed results to {out_p.resolve()}")

    json_path = Path("leaderboard.json")
    md_path = Path("LEADERBOARD.md")
    update_leaderboard_files(entry, json_path, md_path)


if __name__ == "__main__":
    main()
