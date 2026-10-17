# Google - The Gemma 4 Developer Agent Competition — Data

**Competition:** [Google - The Gemma 4 Developer Agent Competition](https://www.kaggle.com/competitions/gemma-4-developer-agent)

**Host:** Google DeepMind

**Competition description:** Accelerate research in autonomous coding agents.

**License:** Apache 2.0

## Dataset Description

This competition evaluates AI software engineering agents on their ability to solve real-world Python bug fixes and feature requests.

To assist agents in large codebase navigation, dependency understanding, and semantic retrieval, the competition provides code graphs, graph/node embeddings, and specialized graph tools alongside traditional repository exploration mechanisms.

Participants are provided with a training set, including tasks, code graphs, and embeddings, for local agent development, prompt engineering, and offline validation.

Submissions are evaluated against a test set curated using a similar pipeline, with identical graph generation methods and verification standards.

In this public-facing version of the dataset, only the training set is available. When a submission is scored, the training set is replaced with the hidden test set.

## Files

### 1. `tasks.jsonl`

A line-delimited JSON file containing the 129 public development tasks.

Each JSON object represents one bug-fixing benchmark instance and includes the reference fix (`patch`) and verification unit tests (`test_patch`) for local evaluation.

#### Fields

| Field | Description |
|---|---|
| `instance_id` | Unique benchmark task identifier combining the repository short name and issue or pull request number. Examples: `fastapi_11194`, `rich_3454`. |
| `repo` | Source GitHub repository in `owner/repo` format. Examples: `fastapi/fastapi`, `Textualize/rich`, `psf/requests`, `encode/httpx`. |
| `base_commit` | 40-character hexadecimal Git commit SHA representing the repository snapshot immediately before the bug fix. |
| `problem_statement` | Natural-language bug report or feature issue description provided to the agent as its task prompt. |
| `hints_text` | Optional issue comments or maintainer hints providing additional context. May be empty. |
| `patch` | Reference unified `git diff` solution patch that resolves the issue on `base_commit`. Provided in `published/tasks.jsonl` for local training and analysis; excluded from the hidden `rerun/tasks.jsonl` test set. |
| `test_patch` | Unified `git diff` patch containing verification unit tests executed by `pytest` during Phase 2 grading. Provided in `published/tasks.jsonl` for local verification; kept in the private `secret/solution.parquet` archive during competition scoring. |
| `created_at` | ISO 8601 timestamp indicating when the issue or pull request was created. |

### 2. `snapshots/`

Directory containing 129 compressed Git working-tree archives.

Each archive represents the repository state at the task's `base_commit`.

#### Structure

```text
snapshots/
├── fastapi_11194.tgz
├── requests_7205.tgz
├── rich_2725.tgz
└── httpx_3672.tgz
```

The filenames follow the pattern:

```text
snapshots/<instance_id>.tgz
```

These snapshots allow agents to inspect and modify the actual source code associated with each issue.

### 3. `graphs/`

Directory containing code graphs generated from the repository snapshots.

The graphs represent source-code symbols and their structural relationships, supporting code navigation, dependency analysis, and retrieval.

#### Graph structure

A graph contains nodes and directed edges.

**Node fields**

| Field | Description |
|---|---|
| `nodes[*].name` | Human-readable qualified symbol name matching `id`. |
| `nodes[*].text` | Full Python source code definition of the function, method, or class at `base_commit`. |

**Edge fields**

| Field | Description |
|---|---|
| `edges` | List of directed structural relationship objects between code nodes. |
| `edges[*].source` | Origin symbol `id` where the call or reference originates. |
| `edges[*].target` | Destination symbol `id` being called, imported, or referenced. |
| `edges[*].type` | Structural relationship category linking `source` to `target`, such as `calls`. |
| `edges[*].key` | Integer index (`0, 1, ...`) distinguishing parallel edges in the multigraph. |

The graphs can be used to identify relevant functions, trace dependencies, and retrieve related code without relying exclusively on text-based search.

### 4. `embeddings/`

Directory containing compressed NumPy archives used for semantic code retrieval.

The dataset contains 256 files: 129 task-named files (`<instance_id>.npz`) hard-linked to 127 commit-named files (`<repo_short>_<base_commit>.npz`).

These archives power offline semantic code retrieval through the `search_similar_code` tool.

#### Embedding structure

Each node embedding is stored as a NumPy array:

```text
<node_id>.npy
```

The array has the following characteristics:

- **Data type:** `float32`
- **Vector length:** 256
- **Feature dimensions:** `embedding_[0-255]`
- **Purpose:** Dense semantic representation of the corresponding AST graph symbol.

Embeddings enable agents to locate functions and methods semantically related to a natural-language issue or a code-related query.

### 5. `HARNESS_README.md`

Documentation describing the competition harness, submission format, and execution environment.

Consult this file for the detailed requirements governing agent configuration, tool usage, execution, and submission packaging.

### 6. Additional input directories

The competition notebook's input listing also shows the following resources:

| Directory | General purpose |
|---|---|
| `docker/` | Docker-related execution and environment resources. |
| `sandbox/` | Sandboxed execution resources. |
| `wheels/` | Python package wheels for the offline environment. |
| `sample_submission/` | Sample submission resources. |

The exact contents of these directories should be inspected in the Kaggle dataset or notebook environment.

## Predefined Agent Tools

The competition harness provides specialized tools for code-graph navigation and semantic retrieval.

### `search_similar_code`

```python
search_similar_code(query: str, k: int = 10) -> str
```

Finds the top-`k` graph nodes with the highest cosine similarity to a query using precomputed embeddings.

**Typical use cases:**

- Find functions relevant to a natural-language bug report.
- Identify code locations related to a feature request.
- Locate potentially relevant implementations before inspecting the repository in detail.

### `get_code_subgraph`

```python
get_code_subgraph(nodes: list[str]) -> str
```

Extracts the induced subgraph, including the nodes and interconnecting edges, for a list of symbols.

**Typical use cases:**

- Trace relationships among candidate functions.
- Examine dependencies and call relationships.
- Understand how a suspected code location interacts with the rest of the repository.

See `HARNESS_README.md` for the detailed tool contracts and execution-environment requirements.

## Dataset Size

The competition notebook's input listing reports approximately **23.3 GB** of input data.

The complete set includes task descriptions, repository snapshots, code graphs, embeddings, sandbox resources, Python package wheels, and supporting documentation.

## Training and Evaluation Workflow

1. Load a task from `tasks.jsonl`.
2. Read the task's `problem_statement`, repository identifier, and `base_commit`.
3. Inspect the corresponding repository snapshot under `snapshots/`.
4. Use code graphs and embeddings to locate relevant functions and understand dependencies.
5. Develop a patch that addresses the reported issue.
6. Validate the patch locally using the available reference patch and verification tests.
7. Package the agent configuration according to the competition's submission requirements.
8. Submit the agent for evaluation against the hidden test set.

## Important Evaluation Notes

- The public development set contains 129 tasks.
- Reference fixes and verification tests are available for local development in the published task data.
- The hidden evaluation set is separate from the public development set.
- During competition scoring, the public training data is replaced with the hidden test data.
- The hidden test set contains approximately 120 tasks, evenly divided between public and private splits according to the published dataset description.
- The hidden test tasks were curated from private repositories.
- The evaluation pipeline applies patches and runs verification tests to determine whether each issue is solved.

## Official Resources

- [Competition overview](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview)
- [Competition data](https://www.kaggle.com/competitions/gemma-4-developer-agent/data)
- [Competition code notebooks](https://www.kaggle.com/competitions/gemma-4-developer-agent/code)
- [Getting Started notebook](https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent)
- [Gemma 4 model](https://www.kaggle.com/models/google/gemma-4)

---

**Source note:** This document reconstructs the publicly indexed Data tab and its associated Getting Started notebook. It is not a byte-for-byte export of the live page. Directory contents and implementation details not exposed by the indexed sources should be verified in `HARNESS_README.md` and the actual competition files.
