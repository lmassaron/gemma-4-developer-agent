# Google - The Gemma 4 Developer Agent Competition

**Host:** Google DeepMind  
**Type:** Featured Prediction Competition  
**Prize Pool:** $65,000  
**Tags:** Artificial Intelligence, Coding  
**Evaluation:** Custom Metric

**Official link:** https://www.kaggle.com/competitions/gemma-4-developer-agent

## Overview

### Accelerate research in autonomous coding agents

Today’s best coding agents rely on massive API models running in the cloud. What if every developer could rely on an autonomous agent equally capable offline, on consumer hardware?

Post-train an open model into a reliable agent that navigates complex codebases and drafts fixes for real software issues, accelerating developer workflows on everyday hardware.

## Description

**Note:** There are two active competitions for the Gemma 4 Developer Agent Competition: this competition, and a paper track, where you can document your approach for this prediction competition.

Today’s most capable coding agents rely on internet-connected models that require powerful commercial infrastructure. That puts advanced software engineering out of reach for many developers.

Small local models have made rapid progress, but they still struggle with complex, multi-turn software engineering. Large codebases are hard to navigate, so fixing bugs and adding features reliably is tough for models that fit on a single accelerator.

In this competition, you’ll post-train Gemma 4 to build autonomous software engineering agents designed to amplify developers’ talents. You’ll explore new fine-tuning and reinforcement learning techniques to create agents that can independently navigate repositories and draft solutions, freeing up human engineers to focus on architecture, logic, and review.

Your solution could help put powerful coding assistants in the hands of millions more developers. Sharing how you did it makes advanced coding tools accessible to everyone.

## Evaluation

You must provide a ZIP archive (`submission.zip`) that contains your Agent Config, comprising system prompts, custom tools, skills, and LoRA adapters.

An `agent.yaml` file must be located at the root of the archive.

### Required submission structure

```text
submission.zip
├── agent.yaml                  # REQUIRED: Root agent config
├── configs/
│   └── sampling.yaml           # Optional: Generation parameters loaded via !include
├── prompts/
│   ├── system.md               # Optional: System instructions loaded via !include
│   └── analyzer.md
├── sub_agents/
│   └── code_analyzer.yaml      # Optional: Sub-agent or AgentTool YAML configurations
├── adapters/                   # Optional: Fine-tuned PEFT LoRA adapters or model weights
│   ├── main_lora/
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   └── tool_lora/
│       ├── adapter_config.json
│       └── adapter_model.safetensors
└── skills/                     # Optional: ADK Skill directories
    └── repo_navigation/
        ├── SKILL.md
        ├── scripts/            # Python or bash scripts executed in a sandbox
        └── resources/          # Domain knowledge markdown files
```

The config language follows the Google ADK Agent Config specification with additional restrictions to prevent code execution outside of a sandbox.

The submissions themselves are compiled into ADK agents to be evaluated.

## Issue Scoring

Similar to SWE-Bench, your agent's submitted patches are evaluated using a PASS/FAIL metric.

For each issue:

1. The submitted patch is applied to the issue's code repository.
2. The repository's validation tests are run.
3. The task passes if the required tests pass.

Your submission's overall score is the percentage of patched repositories that pass the validation tests.

Your agent has a limit of **12 hours** to submit patches for all tasks, inclusive of sandbox setup time, but excluding time for patch validation.

You may, but are not required to, set per-task time limits in your submission's `eval_config.yaml`.

## Timeline

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if necessary.

- **September 23, 2026:** Start date.
- **November 12, 2026:** Optional research paper submission deadline.
- **November 25, 2026:** Entry deadline. You must accept the competition rules before this date to compete.
- **November 25, 2026:** Team merger deadline. Last day participants may join or merge teams.
- **December 2, 2026:** Final submission deadline.

## Prizes

**Total prizes for the main competition: $65,000**

- **1st Place:** $37,000
- **2nd Place:** $18,000
- **3rd Place:** $10,000

### Optional Paper Submission Award

An additional **$35,000** is available for the separate research paper track.

Participants are encouraged to submit original, unpublished research on topics including, but not limited to:

- **Tuning and Optimization:** Parameter-efficient fine-tuning and reinforcement learning (RL) approaches tailored to improve software engineering agent capabilities.
- **Code Comprehension:** Novel techniques for code-graph generation, parsing, and embedding to enhance deep comprehension of complex repositories.
- **Tasks and Benchmarks:** Development of new tasks, evaluation datasets, or resources targeting code structure, graphs, and structured code generation.
- **Graph Reasoning:** Methods and architectures for improving large language model (LLM) reasoning over large-scale graphs.

Top submissions will be highlighted at a Google-hosted event, such as a NeurIPS expo workshop, and awarded prizes.

Submissions leveraging the accompanying released graph and embedding dataset are highly encouraged, but not required.

Paper submissions must be made by **November 12, 2026, at 11:59 PM UTC**.

To submit, sign up for the paper track:

https://www.kaggle.com/competitions/gemma-4-developer-agent-paper

## Model Selection, Budget, and Harness Rules

### Model Selection and LoRA Adapters

At present, only the following Gemma 4 model variant is supported:

```text
gemma-4-31b-it-qat-w4a16-ct
```

You must choose this model for every agent and subagent.

You may, but are not required to, include one or more LoRA adapters in `.safetensors` format with your submission.

Although you are restricted to a single base model, you may use different adapters for each agent in your submission.

To use LoRA adapters:

1. Place PEFT LoRA directories containing `adapter_config.json` and `adapter_model.safetensors` inside `adapters/<adapter_name>/`.
2. Reference the adapter using `adapter: <adapter_name>` on any `LlmAgent` in `agent.yaml` or `sub_agents/*.yaml`.

For example, use `adapter: main_lora` on the root coder agent and `adapter: tool_lora` on a read-only analyzer `AgentTool`.

### Tool, Prompt, and Skill Rules

#### `!include` directives

The `!include` directive resolves file paths relative to the directory containing the file with the directive.

For example, in `agent.yaml`:

```yaml
!include prompts/system.md
```

This loads `prompts/system.md`.

#### Sandboxing

Path traversal outside the submission root is not allowed. This includes traversal through paths such as `../` and through symbolic links.

#### Allowed tools

Your agent can only request tools provided by the competition harness or custom subagents defined via `agent_tool`.

#### Skill structure and sandboxing

Each skill must be a directory containing a `SKILL.md` manifest with YAML frontmatter, including:

```yaml
name: <skill-name>
```

Scripts executed via `run_skill_script` run securely inside the competition's persistent Docker container.

They share filesystem access with `run_command` and debit execution time against your central budget.

Agents can inspect domain knowledge files using `load_skill_resource`.

#### Predefined tools

The competition harness provides tools including:

```python
search_similar_code(query: str, k: int = 10) -> str
get_code_subgraph(nodes: list[str]) -> str
```

- `search_similar_code` finds the top-k graph nodes with the highest cosine similarity to a query in the precomputed embeddings.
- `get_code_subgraph` extracts the induced subgraph, including nodes and interconnecting edges, for a list of symbols.

See `HARNESS_README.md` in the dataset for detailed information about the submission format and execution environment.

### Upgraded Accelerators

This competition has access to Kaggle's pool of powerful new L4x4 machines.

These machines offer **96 GB of GPU memory**, enabling submissions with much larger models.

#### Quota usage

Due to limited availability, notebooks using L4x4 machines consume GPU quota at twice the rate of older T4x2 and P100 machines.

Kaggle may increase this rate as necessary to ensure the machines remain available for submission scoring.

#### Restricted use

L4 machines are available only for notebooks attached to this competition.

Attempts to circumvent this restriction may be enforced by Kaggle moderation, with consequences including team bans or account bans.

#### Internet access

All L4 sessions must have internet disabled.

## Competition Host

Google DeepMind

## Participation

Participation figures displayed by Kaggle may change over time.

## Tags

- Artificial Intelligence
- Coding

## Citation

Elan Markowitz, Bryan Perozzi, Benedek Rózemberczki, Glenn Cameron, Hadi Hemmati, Yuchen Li, Michael Galkin, Majid Farhadi, Ryan Holbrook, and Ashley Oldacre. *Google - The Gemma 4 Developer Agent Competition*. Kaggle, 2026.

https://www.kaggle.com/competitions/gemma-4-developer-agent
