# Gemma 4 Developer Agent — Local Benchmark Leaderboard

> Standardized evaluation of Gemma 4 developer agent candidates on verified SWE-bench tasks under official 360s/task competition constraints.

## Overall Leaderboard

| Rank | Model / Configuration | Suite | Resolution Rate | Resolved | Avg Time / Task | Timeouts (>360s) | Avg Turns | Last Evaluated |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| #1 | **v19_hill_climbed** | `climb_12_cross` | **66.7%** | 8/12 | 237.9s | 4 | 9.0 | 2026-10-29 21:14:26 |
| #2 | **v16_phased_edit_enforcement** | `smoke_3` | **66.7%** | 2/3 | 215.5s | 0 | 13.0 | 2026-10-28 19:01:54 |
| #3 | **v17_hill_climbed** | `climb_11` | **63.6%** | 7/11 | 158.7s | 1 | 8.4 | 2026-10-29 07:28:00 |
| #4 | **v18_hill_climbed** | `climb_12_expanded` | **50.0%** | 6/12 | 197.2s | 2 | 9.1 | 2026-10-29 13:26:00 |
| #5 | **v17_hill_climbed** | `climb_12_expanded` | **50.0%** | 6/12 | 225.4s | 3 | 10.3 | 2026-10-29 09:05:00 |
| #6 | **v35_compaction2048_lean6_lora** | `core_12` | **50.0%** | 6/12 | 438.6s | 6 | 12.6 | 2026-11-11 12:02:58 |
| #7 | **v18_hill_climbed** | `external_hf_py312` | **41.7%** | 5/12 | 346.8s | 6 | 12.3 | 2026-10-29 15:46:00 |
| #8 | **v34_clean024_mandatory_verify_anchor** | `core_12` | **41.7%** | 5/12 | 416.6s | 6 | 12.4 | 2026-11-11 10:26:45 |
| #9 | **v19_hill_climbed** | `core_60` | **23.3%** | 14/60 | 294.0s | 38 | 11.3 | 2026-10-30 01:50:22 |
| #10 | **v16_phased_edit_enforcement** | `core_60` | **20.0%** | 12/60 | 350.9s | 33 | 12.6 | 2026-10-29 01:35:00 |
| #11 | **gemma-4-12b-sft-v6** | `smoke_3` | **0.0%** | 0/3 | 146.3s | 0 | 20.0 | 2026-10-19 08:47:43 |
| #12 | **v10_lean_thinking** | `smoke_3` | **0.0%** | 0/3 | 269.2s | 1 | 15.0 | 2026-10-25 11:08:51 |

## Benchmark Suites
- **`smoke_3`**: 3 representative tasks across core repositories (`psf/requests`, `fastapi/fastapi`, `Textualize/rich`) for rapid timeout verification and regression testing.
- **`core_12`**: 12 balanced tasks (4 Requests, 4 FastAPI, 4 Rich) providing a high-confidence local signal.

## Latest Evaluation Run Details
**Run ID**: `v35_compaction2048_lean6_lora_core_12_2026-11-11 12:02:58`  
**Model**: `v35_compaction2048_lean6_lora` | **Suite**: `core_12`  

| Instance ID | Repository | Status | Resolved | Duration (s) | Turns | Tool Calls | Patch Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `requests_7205` | `psf/requests` | `RESOLVED` | ✅ PASS | 481.17s | 14 | 14 | 928 B |
| `requests_7433` | `psf/requests` | `RESOLVED` | ✅ PASS | 162.18s | 8 | 8 | 584 B |
| `requests_7502` | `psf/requests` | `RESOLVED_ANYTIME` | ✅ PASS | 480.19s | 11 | 11 | 615 B |
| `requests_7505` | `psf/requests` | `RESOLVED` | ✅ PASS | 234.68s | 12 | 12 | 1049 B |
| `fastapi_11194` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 533.53s | 14 | 14 | 686 B |
| `fastapi_11355` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 497.65s | 19 | 19 | 0 B |
| `fastapi_12942` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 502.1s | 22 | 22 | 0 B |
| `fastapi_13537` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 609.72s | 8 | 8 | 0 B |
| `rich_2725` | `Textualize/rich` | `TIMEOUT` | ❌ FAIL | 543.68s | 7 | 7 | 0 B |
| `rich_2943` | `Textualize/rich` | `RESOLVED` | ✅ PASS | 349.79s | 13 | 13 | 400 B |
| `rich_3006` | `Textualize/rich` | `RESOLVED` | ✅ PASS | 326.64s | 15 | 15 | 558 B |
| `rich_3052` | `Textualize/rich` | `TIMEOUT` | ❌ FAIL | 542.19s | 8 | 8 | 1964 B |
