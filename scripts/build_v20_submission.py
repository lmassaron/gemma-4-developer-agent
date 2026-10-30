#!/usr/bin/env python3
"""Build and validate the v20 Kaggle submission bundle.

v20 = community "Budget-Fit" configuration (8 min / 28 calls / 240 s / thinking 4096 /
temperature 0.2, graph tools unregistered) + a general, task-agnostic "anytime patch"
prompt. No benchmark-specific hints.

The validator also blocks the failure that zeroed v16: Google ADK treats any
`{identifier}` in the instruction as a session-state placeholder and raises
`KeyError: Context variable not found` on every model call when it is missing.
"""
import hashlib
import re
import shutil
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "state_of_the_art" / "bundle_v20"
ARCHIVE = ROOT / "submissions_archive" / "submission_v20_budget_fit_anytime.zip"
ROOT_ZIP = ROOT / "submission.zip"

FILES = ["agent.yaml", "configs/sampling.yaml", "eval_config.yaml", "prompts/system.md"]
ALLOWED_TOOLS = {"run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch"}
GRAPH_TOOLS = {"get_code_neighbors", "search_similar_code", "get_code_subgraph"}
# Same pattern ADK uses in google/adk/flows/llm_flows/prompt/_instructions_utils.py
ADK_TEMPLATE_VAR = re.compile(r"(?<![\$\{\\]){+[^{}]*}+")


def adk_placeholders(text: str) -> list[str]:
    """Return placeholders that ADK would try to resolve from session state."""
    hits = []
    for m in ADK_TEMPLATE_VAR.finditer(text):
        name = m.group().lstrip("{").rstrip("}").strip().rstrip("?")
        if name.startswith("artifact.") or name.isidentifier() or (
            ":" in name and name.split(":", 1)[1].isidentifier()
        ):
            hits.append(m.group())
    return hits


def validate(bundle: Path) -> list[str]:
    problems = []
    present = sorted(str(p.relative_to(bundle)) for p in bundle.rglob("*") if p.is_file())
    # Allow optional adapters under adapters/<name>/...
    non_adapter_files = [f for f in present if not f.startswith("adapters/")]
    if sorted(non_adapter_files) != sorted(FILES):
        problems.append(f"bundle files {non_adapter_files} != {sorted(FILES)}")

    agent = yaml.safe_load((bundle / "agent.yaml").read_text().replace("!include ", ""))
    if agent.get("model") != "gemma-4-31b-it-qat-w4a16-ct":
        problems.append(f"unexpected model {agent.get('model')}")
    tools = set(agent.get("tools", []))
    if tools - ALLOWED_TOOLS:
        problems.append(f"unexpected tools {sorted(tools - ALLOWED_TOOLS)}")
    if tools & GRAPH_TOOLS:
        problems.append("graph tools must stay unregistered")

    samp = yaml.safe_load((bundle / "configs/sampling.yaml").read_text())
    allowed_samp = {"temperature", "top_p", "top_k", "max_output_tokens", "thinking_config"}
    if set(samp) - allowed_samp:
        problems.append(f"non-ADK sampling keys {sorted(set(samp) - allowed_samp)}")
    if not samp.get("thinking_config", {}).get("include_thoughts"):
        problems.append("include_thoughts must be true (ablation: -16 tasks when off)")

    ev = yaml.safe_load((bundle / "eval_config.yaml").read_text())
    if "evaluation" not in ev:
        problems.append("eval_config.yaml must nest limits under `evaluation:`")

    prompt = (bundle / "prompts/system.md").read_text()
    ph = adk_placeholders(prompt)
    if ph:
        problems.append(f"ADK would treat these as state placeholders (KeyError at runtime): {ph}")
    if re.search(r"\bcall:\w+\{", prompt):
        problems.append("prompt contains literal call:tool{...} syntax")
    for t in GRAPH_TOOLS:
        if t in prompt:
            problems.append(f"prompt names unregistered tool {t}")
    return problems


def build() -> None:
    problems = validate(BUNDLE)
    if problems:
        print("BLOCKING:")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE, "w") as zf:
        for name in FILES:  # file entries only, no directory entries
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            zf.writestr(info, (BUNDLE / name).read_bytes())
    with zipfile.ZipFile(ARCHIVE) as zf:
        assert zf.testzip() is None and sorted(zf.namelist()) == sorted(FILES)
    shutil.copyfile(ARCHIVE, ROOT_ZIP)
    print("OK:", ARCHIVE.relative_to(ROOT), "->", ROOT_ZIP.name)
    print("SHA-256:", hashlib.sha256(ARCHIVE.read_bytes()).hexdigest())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        for z in sys.argv[2:]:
            tmp = Path("/tmp/_bundle_check") / Path(z).stem
            shutil.rmtree(tmp, ignore_errors=True)
            zipfile.ZipFile(z).extractall(tmp)
            print(z, "->", validate(tmp) or "OK")
    else:
        build()

