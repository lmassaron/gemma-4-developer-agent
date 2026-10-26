#!/usr/bin/env python3
"""
Inspect data/tasks.jsonl and list unique repositories, commits, and candidate 60-task suite.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent.parent
    tasks_file = root / "data" / "tasks.jsonl"
    assert tasks_file.is_file(), f"Tasks file not found: {tasks_file}"

    tasks = []
    repo_counts = Counter()
    repo_tasks = defaultdict(list)

    with open(tasks_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            t = json.loads(line)
            tasks.append(t)
            repo = t.get("repo", "unknown")
            repo_counts[repo] += 1
            repo_tasks[repo].append(t)

    print(f"Total tasks loaded: {len(tasks)}")
    print("\nTasks per repository:")
    for repo, count in repo_counts.most_common():
        print(f"  {repo:<30}: {count:>3} tasks")

    # Select a balanced 60-task suite
    # Proportionally pick tasks from each repository
    suite_60 = []
    # Determine target per repo
    for repo, t_list in repo_tasks.items():
        # take up to ~50% from each repo to reach 60
        target = max(1, int(len(t_list) * (60 / len(tasks))))
        selected = t_list[:target]
        suite_60.extend([x["instance_id"] for x in selected])

    # Fill up or trim to exactly 60
    all_instance_ids = [x["instance_id"] for x in tasks]
    for tid in all_instance_ids:
        if len(suite_60) >= 60:
            break
        if tid not in suite_60:
            suite_60.append(tid)
    suite_60 = suite_60[:60]

    print(f"\nBalanced 60-task suite selected ({len(suite_60)} tasks):")
    print(json.dumps(suite_60, indent=2))

    out_file = root / "state_of_the_art" / "benchmarks_data" / "test_tasks_core_60.json"
    out_file.write_text(json.dumps({"suite_name": "core_60", "tasks": suite_60}, indent=2))
    print(f"\nSaved suite specification to {out_file}")

if __name__ == "__main__":
    main()
