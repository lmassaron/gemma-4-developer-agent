# Gemma 4 Developer Agent

Code, experiments, and submission pipeline for the Kaggle [Google - The Gemma 4 Developer Agent Competition](https://www.kaggle.com/competitions/gemma-4-developer-agent).

## Overview

The competition evaluates autonomous AI software engineering agents on their ability to resolve real-world Python issues (SWE-bench style) using the `gemma-4-31b-it-qat-w4a16-ct` model within a 12-hour global budget.

## Repository Structure

```text
├── submission/                      # Declarative agent configuration bundle
│   ├── agent.yaml                   # Root ADK agent configuration
│   ├── configs/
│   │   └── sampling.yaml            # Sampling parameters & thinking budget
│   ├── eval_config.yaml             # Per-task execution budget & turn limits
│   └── prompts/
│       └── system.md                # Contract-first prompt with anti-pattern avoidance
├── scripts/
│   ├── package_and_validate.py      # Validates against harness specs & builds submission.zip
│   ├── evaluate_local.py            # Local SWE-bench benchmark & evaluation harness
│   └── download_gemma_model.py      # Downloads Gemma 4 model variants using KaggleHub
├── documents/                       # Competition guidelines, dataset docs, and reference notebooks
├── submission.zip                   # Verified submission archive ready for Kaggle
└── .gitignore
```

## Quick Start

### 1. Environment Setup (using `uv`)

```bash
uv venv .venv
source .venv/bin/activate
uv pip install kaggle kagglehub pytest pyyaml
```

### 2. Download Competition Dataset

```bash
python3 -c "
import kagglehub
path = kagglehub.competition_download('gemma-4-developer-agent')
print('Data downloaded to:', path)
"
ln -s ~/.cache/kagglehub/competitions/gemma-4-developer-agent data
```

### 3. Package & Validate Submission

Validates the submission bundle against official harness rules (single model check, no symlinks, under 3 GiB, deterministic ZIP format):

```bash
python3 scripts/package_and_validate.py
```

### 4. Local SWE-bench Benchmark Evaluation

Run offline verification and task soundness checks on the public development tasks:

```bash
# Evaluate reference gold fixes on benchmark tasks:
python3 scripts/evaluate_local.py --repo psf/requests --limit 5 --mode gold-eval

# Audit task soundness (tests fail before fix, pass after fix):
python3 scripts/evaluate_local.py --task-id requests_7205 --mode audit
```

### 5. Download Gemma 4 Models

```bash
# Competition 31B QAT model:
python3 scripts/download_gemma_model.py --model google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct

# Lightweight model for local fast testing:
python3 scripts/download_gemma_model.py --model google/gemma-4/other/gemma-4-e2b-it-qat-w4a16-ct
```

### 6. Submit to Kaggle

```bash
kaggle competitions submit -c gemma-4-developer-agent -f submission.zip -m "Contract-first single agent with thinking enabled"
```
