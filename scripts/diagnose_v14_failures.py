import json
import re

with open('state_of_the_art/benchmarks_data/benchmark_v14_clean_60_results.json') as f:
    v14_data = json.load(f)['v14_heredoc_synthesis']['tasks']

flex_results = []
with open('state_of_the_art/benchmarks_data/bench_flexonafft_vs_v14.log') as f:
    for line in f:
        if 'Result ->' in line and 'Status:' in line:
            m_res = re.search(r'Resolved:\s*(True|False)', line)
            m_stat = re.search(r'Status:\s*(\w+)', line)
            m_dur = re.search(r'Duration:\s*([\d\.]+)s', line)
            m_turns = re.search(r'Turns:\s*(\d+)', line)
            if m_res and m_stat:
                flex_results.append({
                    'resolved': m_res.group(1) == 'True',
                    'status': m_stat.group(1),
                    'duration': float(m_dur.group(1)) if m_dur else 0.0,
                    'turns': int(m_turns.group(1)) if m_turns else 0,
                })
flex_results = flex_results[:60]

regressions = []
both_passed = []
categories = {
    '[A] 0-patch / timeout / max_tool_calls exhausted': 0,
    '[B] edit_file / heredoc whitespace/match failure loop': 0,
    '[C] vLLM context length / tool parsing stall': 0,
    '[D] wrong file / location edited': 0,
    '[E] partial fix / passed repro but failed hidden tests': 0,
    '[F] dirty working tree / test file modified': 0,
}

for i, v in enumerate(v14_data):
    tid = v['instance_id']
    repo = v['repo']
    f_res = flex_results[i]['resolved'] if i < len(flex_results) else False
    v_res = v['resolved']
    v_status = v['status']
    v_calls = v.get('tool_calls', [])
    v_snip = v.get('output_snippet', '')

    if f_res and not v_res:
        regressions.append((i+1, tid, repo))
    elif f_res and v_res:
        both_passed.append((i+1, tid, repo))
    else:
        # Categorize failures
        has_edit_attempt = any('old in c' in str(c) or 'edit_file' in str(c) or 'write_file' in str(c) for c in v_calls)
        has_no_tool = any('[NO_TOOL]' in str(c) for c in v_calls)
        
        if v_status == 'TIMEOUT' or v['turns'] >= 28:
            categories['[A] 0-patch / timeout / max_tool_calls exhausted'] += 1
        elif has_no_tool:
            categories['[C] vLLM context length / tool parsing stall'] += 1
        elif has_edit_attempt:
            # Check if edit failed to apply or if it applied but failed test
            if 'AssertionError: Target block not found' in str(v_calls) or v['patch_size'] == 0:
                categories['[B] edit_file / heredoc whitespace/match failure loop'] += 1
            else:
                categories['[E] partial fix / passed repro but failed hidden tests'] += 1
        elif 'ModuleNotFoundError' in v_snip or 'Unknown config option' in v_snip:
            categories['[D] wrong file / location edited'] += 1
        else:
            categories['[A] 0-patch / timeout / max_tool_calls exhausted'] += 1

print(f"Total tasks analyzed: {len(v14_data)}")
print(f"Regressions (flexonafft_v2 passed, v14 failed): {len(regressions)}")
print(f"Both passed: {both_passed}")
print("\n=== FAILURE MODE CATEGORIZATION (59 non-resolved tasks) ===")
for cat, count in categories.items():
    pct = (count / 59) * 100
    print(f"{cat}: {count} tasks ({pct:.1f}%)")
