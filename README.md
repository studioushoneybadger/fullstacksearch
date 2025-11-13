# X Engineer Candidate Funnel and Evaluator

This program uses the Grok API (from xAI) to identify and evaluate potential full-stack engineer candidates on X (formerly Twitter). It searches for users based on 9 predefined criteria related to software engineering skills, filters them, and then grades them using AI analysis. The process is divided into two stages: funneling candidates and grading/summarizing.

## Overview

- **Wrapper (wrapper.py)**: Runs Stage 1 and, if successful, automatically runs Stage 2. It prompts for your Grok API key.
- **Stage 1 (_1_funnel.py)**: Searches X for users matching criteria like end-to-end ownership, modern stack mastery, AI integration, etc. It verifies handles, applies filters (e.g., follower count, account age), and outputs a list of candidates.
- **Stage 2 (_2_grade_and_summarize.py)**: Takes the candidates from Stage 1, evaluates them against the criteria using Grok API, calculates scores based on matches, engagement, and recency, and ranks the top candidates.


The program aims to find high-quality candidates by running multiple searches per criterion, excluding duplicates, and verifying results to avoid hallucinations.

## Requirements

- Python 3.8+ (tested on 3.12)
- Grok API key (from xAI Premium+ subscription, as it uses rate-limited endpoints)
- Internet access for API calls
- Installed libraries (install via `pip` if missing):
  - `requests`
  - `tabulate` (for console tables)
  - Other standard libraries: `json`, `os`, `time`, `subprocess`, `threading`, `concurrent.futures`, `csv`, `datetime`, `collections`, `re`, `random`, `logging`

No additional package installations are needed beyond these (the scripts use built-in or common modules).

## Setup

1. Clone or download the repository.
2. Ensure you have a valid Grok API key (from https://x.ai or your xAI account).
3. Place all three scripts (`wrapper.py`, `_1_funnel.py`, `_2_grade_and_summarize.py`) in the same directory.

## Usage

1. Run the wrapper script:
   ```
   python wrapper.py
   ```
2. When prompted, enter your Grok API key.
3. The script will:
   - Run Stage 1: Search and funnel candidates (outputs `step1_funneled_candidates.json`).
   - If "DONEZO" is printed (indicating success), run Stage 2: Grade and summarize (outputs `step2_raw_grades.json`, `_15fullstackengineers.json`, and `_15potential_fullstack_candidates.csv`).
4. Monitor the console for progress (e.g., API calls, verification, rankings).

### Configuration

- Edit `_1_funnel.py` for search tweaks (e.g., `TARGET_HANDLES`, `RUNS_PER_CRITERION`, filters like min followers).
- Edit `_2_grade_and_summarize.py` for evaluation settings (e.g., `TOP_N_RESULTS`, scoring weights).
- Debug mode: Set `DEBUG = True` in `_2_grade_and_summarize.py` or `DEBUG_MODE = True` in `_1_funnel.py` to print prompts/responses.

### Output Files

- `step1_funneled_candidates.json`: List of candidate handles from Stage 1.
- `step2_raw_grades.json`: Detailed evaluations, scores, and rankings from Stage 2.
- `_15fullstackengineers.json`: Top 15 ranked candidates (summary).
- `_15potential_fullstack_candidates.csv`: CSV of top 15 handles.

## Notes

- **Rate Limiting**: The scripts respect Grok API limits (160 calls/min, 400k tokens/min). Adjust if needed.
- **Verification**: Stage 1 verifies handles with retries to reduce false positives.
- **Customization**: Criteria are defined in the scripts—modify queries or instructions for different roles.
- **Limitations**: Relies on public X data; results may vary. API costs apply based on your xAI plan.
- **Troubleshooting**: Check console logs for errors (e.g., rate limits, invalid API key). Increase delays if throttled.

For questions or improvements, feel free to contribute!
