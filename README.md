# fullstacksearch
using Grok API to search for fullstack software engineers.  checks for authenticity and ranks them based on 9 criteria.


# Full-Stack Engineer Candidate Finder on X

This Python script automates the process of identifying and evaluating potential full-stack engineers on X (formerly Twitter) using the Grok API from xAI. It consists of two main phases:

1. **Funneling Candidates**: Searches for X users matching 9 predefined criteria related to full-stack engineering skills (e.g., end-to-end ownership, modern stack mastery, AI integration). It uses semantic or keyword searches, filters out invalid or duplicate handles, and verifies them to ensure they are real accounts.
2. **Grading and Ranking**: Evaluates the funneled candidates against the same criteria by analyzing their profiles, bios, and recent posts. It calculates scores based on matches, engagement, and recency, then ranks the top candidates.

The script runs these phases sequentially: it executes the funneling phase first, checks for successful completion ("DONEZO" in output), and then proceeds to grading.

## Features
- **High-Quality Mode**: Focuses on accurate, non-hallucinated results by excluding previously found handles, applying strict filters (e.g., follower counts, account age), and verifying handles.
- **Rate Limiting**: Built-in rate limiters to respect Grok API limits (calls and tokens per minute).
- **Parallel Processing**: Uses threading for concurrent API calls during evaluation.
- **Output Formats**: Generates JSON files for raw results and summaries, a CSV of top handles, and console tables for easy viewing.
- **Criteria-Based Search**: 9 customizable criteria covering full-stack skills, DevOps, AI, and more.
- **Verification**: Strict validation to ensure only real X accounts are included.

## Requirements
- **Python**: 3.12+ (tested with 3.12.3).
- **Libraries**: Install via `pip install -r requirements.txt` (create the file with the list below if needed):
  ```
  requests
  tabulate
  ```
  (Other imports like `json`, `logging`, `re`, `random`, `collections`, `typing`, `datetime`, `concurrent.futures`, `threading`, `csv`, and `io` are standard library modules.)
- **Grok API Key**: A valid xAI Grok API key (Premium+ subscription recommended for rate limits). Set it in the script under `CONFIG.GROK_API_KEY` and `GROK_API_KEY`.
- **Internet Access**: For API calls to `https://api.x.ai`.
- **No Additional Installs**: The script avoids external package installations beyond what's listed.

## Installation
1. Clone or download the script (`fullstack.py`).
2. Create a `requirements.txt` file with the above libraries and run `pip install -r requirements.txt`.
3. Replace the placeholder API key in the script with your own:
   ```python
   GROK_API_KEY = "your-grok-api-key-here"
   ```
   (Appears in two places: once in the funnel config and once in the grade config.)

## Usage
1. **Run the Script**:
   ```
   python fullstack.py
   ```
   - The script will first run the funneling phase (searching and verifying candidates).
   - If successful (prints "DONEZO"), it will automatically proceed to the grading phase.
   - Press any key to exit after completion (Mac/Unix compatible).

2. **Debug Mode**: Set `CONFIG.DEBUG_MODE = True` to print prompts and debug info.
3. **Customization**:
   - Edit the `CONFIG` class in the funnel section for search parameters (e.g., `HANDLES_PER_RUN`, `RUNS_PER_CRITERION`, filters like `MIN_FOLLOWERS`).
   - Edit grading config (e.g., `DATE_RANGE_DAYS`, `SCORING_WEIGHTS`, `TOP_N_RESULTS`).
   - Modify `CRITERIA` dictionaries for custom search queries and instructions.

4. **Expected Runtime**:
   - Depends on configuration (e.g., 9 criteria × 4 runs × 5 handles = ~180 API calls).
   - With rate limiting, it may take 5-30 minutes. Increase `DELAY_BETWEEN_CALLS` if throttled.

## Configuration Details
### Funnel Phase (`funnel_main()`)
- **API Settings**: URL, model, key.
- **Rate Limits**: Calls/tokens per minute (set to 80% of Premium+ limits).
- **Quantity**: `HANDLES_PER_RUN` (default 5 to avoid hallucinations), `RUNS_PER_CRITERION` (default 4).
- **Filters**: Account age, followers, tweets, bio length, blocked keywords, etc.
- **Criteria**: 9 predefined with query groups, dates, and instructions (semantic/keyword search).

### Grade Phase (`grade_main()`)
- **Search Limits**: Posts per query, date range.
- **Parallelism**: `MAX_CONCURRENT_REQUESTS` (default 5).
- **Scoring**: Weights for matches, engagement, recency; customizable.
- **Input**: Loads candidates from `step1_funneled_candidates.json` (output of funnel phase).
- **Criteria Definitions**: Keywords and descriptions for evaluation.

## Output
- **Console**: Progress logs, verification details, final ranked table of top engineers (handle, score, matches, etc.).
- **Files**:
  - `step1_funneled_candidates.json`: Funneled handles with reasons and criteria.
  - `step2_raw_grades.json`: Raw evaluations, aggregated scores, and config.
  - `_10fullstackengineers.json`: Summary of top 10 ranked engineers.
  - `_10potential_fullstack_candidates.csv`: CSV of top 10 handles (header: "handle").

Example Console Output:
```
✓ @user1 is valid
...
Top 10 Engineers (by score):
1. @engineer1 - Score: 8.5
   Matches (7/9): End-to-End Ownership, ...
...
Results saved to step2_raw_grades.json
```

## Troubleshooting
- **API Errors**: Check key validity, rate limits, or increase `DELAY_BETWEEN_CALLS`/`MAX_RETRIES`.
- **Hallucinations**: Keep `HANDLES_PER_RUN` ≤7; use high-quality mode.
- **No Candidates**: Adjust filters (e.g., lower `MIN_FOLLOWERS`) or broaden queries.
- **Parsing Issues**: Ensure JSON responses are clean; debug with `DEBUG_MODE`.
- **Rate Limiting**: If throttled, reduce `MAX_CONCURRENT_REQUESTS` or adjust limits.

## Limitations
- Relies on Grok API for X searches (may miss some users).
- Verification is strict; false negatives possible.
- No direct X API access; uses Grok's tools (e.g., `x_semantic_search`, `x_user_search`).
- Ethical Note: This is for research/recruitment; respect user privacy and X terms.

## License
This script is provided as-is under the MIT License. Use responsibly.

For questions, contact the maintainer or open an issue if this is in a repo.
