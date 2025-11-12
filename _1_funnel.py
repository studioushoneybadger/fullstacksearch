# File: _1_funnel.py

import subprocess
import time
import sys
import termios
import os
import json
import logging
import re
import requests
import random
from collections import defaultdict, deque
from tabulate import tabulate
from typing import List, Dict, Set
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import csv
from io import StringIO

import os

GROK_API_KEY = os.environ['GROK_API_KEY']

"""
_1_funnel.py – HIGH-QUALITY MODE
• 9 criteria
• 10 best handles per run
• Excludes previously found handles
• Adjustable delay between API calls
• supposedly eliminates hallucinated data.
"""
# ----------------------------------------------------------------------
# USER CONFIGURATION (Tweak here!)
# ----------------------------------------------------------------------
class CONFIG:
    # === API & MODEL ===
    API_URL = "https://api.x.ai/v1/chat/completions"
    MODEL = "grok-4-fast-reasoning"
    # === RATE LIMITING (80% of Premium+) ===
    CALLS_PER_MIN = 160
    TOKENS_PER_MIN = 400_000
    ESTIMATED_TOKENS_PER_CALL = 1800
    # === QUALITY & QUANTITY ===
    HANDLES_PER_RUN = 7 #changing the handles per run to more than 7 is hallucination station.
    RUNS_PER_CRITERION = 3 #this is the main variable to change if you want a larger dataset.
    # === NEW: DELAY BETWEEN API CALLS ===
    DELAY_BETWEEN_CALLS = 1.0 # seconds – increase to be gentler on API / avoid throttling
    # === FILTERS ===. not strictly followed
    MIN_AGE_YEARS = 0.5
    MIN_FOLLOWERS = 50
    MAX_FOLLOWERS = 100_000
    MAX_FOLLOWING_RATIO = 1.0
    MIN_TWEETS = 10
    MIN_BIO_LENGTH = 1
    BLOCKED_KEYWORDS = []#"Inc", "Corp", "LLC", "GmbH", "Company", "Team", "Official", "Bot"]
    FORBIDDEN_VERIFIED_TYPE = "business"
    DEFAULT_PROFILE_IMAGE_CONTAINS = "default_profile"
    # === OUTPUT ===
    DEBUG_MODE = False
    OUTPUT_JSON_FILE = "step1_funneled_candidates.json"
    OUTPUT_TABLE = True
    # === CRITERIA (edit queries, dates, instructions) ===
    CRITERIA = {
        1: {
            "name": "End-to-End Ownership",
            "search_type": "semantic",
            "query_groups": [
                ["end-to-end ownership", "ship solo", "0-to-1 features", "research to production"],
                ["DB", "API", "UI", "deploy", "monitor", "scale"]
            ],
            "since": "2020-01-01",
            "instruction": "Pick authors who describe shipping complete features solo from research through DB, API, UI, deployment, monitoring, and owning them at scale."
        },
        2: {
            "name": "Modern Stack Mastery",
            "search_type": "semantic",
            "query_groups": [
                ["modern stack", "full-stack mastery"],
                ["TypeScript", "Python", "Rust", "React", "Next.js", "FastAPI", "Node", "Docker", "AWS", "GCP"]
            ],
            "since": "2020-01-01",
            "instruction": "Pick authors demonstrating expertise in modern stacks like TypeScript + Python/Rust, React/Next.js + FastAPI/Node, and Docker with cloud providers like AWS/GCP."
        },
        3: {
            "name": "TypeScript Everywhere",
            "search_type": "semantic",
            "query_groups": [
                ["TypeScript everywhere", "fullstack TypeScript"],
                ["frontend", "backend", "tRPC", "NestJS", "zero runtime bugs"]
            ],
            "since": "2020-01-01",
            "instruction": "Pick authors using TypeScript across full-stack (frontend and backend via tRPC/NestJS) with emphasis on eliminating runtime bugs."
        },
        4: {
            "name": "Performance & System Design",
            "search_type": "semantic",
            "query_groups": [
                ["performance optimization", "system design"],
                ["low-latency", "WebSocket", "WebRTC", "scalable pipelines", "p95 <100ms", "caching", "rate limiting"]
            ],
            "since": "2020-01-01",
            "instruction": "Pick authors discussing low-latency systems (WebSocket/WebRTC), scalable pipelines, achieving <100ms p95, caching, and rate limiting."
        },
        5: {
            "name": "DevOps & Reliability",
            "search_type": "semantic",
            "query_groups": [
                ["DevOps", "reliability engineering"],
                ["CI/CD", "GitHub Actions", "observability", "Sentry", "Datadog", "Terraform", "Kubernetes"]
            ],
            "since": "2020-01-01",
            "instruction": "Pick authors experienced in CI/CD (GitHub Actions), observability (Sentry/Datadog), and infrastructure basics like Terraform/K8s."
        },
        6: {
            "name": "Product Thinking",
            "search_type": "semantic",
            "query_groups": [
                ["product thinking", "MVP shipping"],
                ["Figma", "challenge specs", "ship MVP <48h", "validate with users"]
            ],
            "since": "2020-01-01",
            "instruction": "Pick authors who talk about going from Figma designs, challenging specs, shipping MVPs in under 48 hours, and validating with real users."
        },
        7: {
            "name": "Speed & Debugging",
            "search_type": "semantic",
            "query_groups": [
                ["speed optimization", "debugging expertise"],
                ["obsess over ms/bytes", "bisect outages <30 min", "bias for velocity"]
            ],
            "since": "2020-01-01",
            "instruction": "Pick authors obsessed with speed (ms/bytes), bisecting production outages in under 30 minutes, and maintaining a bias for velocity."
        },
        8: {
            "name": "AI/LLM Integration",
            "search_type": "semantic",
            "query_groups": [
                ["AI integration", "LLM hooks"],
                ["Grok", "prompt chaining", "RAG", "function calling", "streaming UX"]
            ],
            "since": "2020-01-01",
            "instruction": "Pick authors with deep integrations of AI/LLMs like Grok, including prompt chaining, RAG, function calling, and streaming UX."
        },
        9: {
            "name": "Domain Research & Pipelines",
            "search_type": "semantic",
            "query_groups": [
                ["domain research", "data pipelines"],
                ["high-throughput systems", "WebRTC", "audio encoding", "vector DBs", "media processing"]
            ],
            "since": "2020-01-01",
            "instruction": "Pick authors building high-throughput media/data systems involving WebRTC, audio encoding, vector DBs, and related pipelines."
        }
    }
# ----------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG if CONFIG.DEBUG_MODE else logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
# ----------------------------------------------------------------------
# RATE LIMITER
# ----------------------------------------------------------------------
class FunnelRateLimiter:
    def __init__(self):
        self.tokens = CONFIG.TOKENS_PER_MIN
        self.calls = CONFIG.CALLS_PER_MIN
        self.last_update = time.time()
        self.lock = Lock()

    def wait(self, token_cost=None):
        if token_cost is None:
            token_cost = CONFIG.ESTIMATED_TOKENS_PER_CALL
        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.last_update
                self.calls += elapsed * (CONFIG.CALLS_PER_MIN / 60.0)
                self.tokens += elapsed * (CONFIG.TOKENS_PER_MIN / 60.0)
                self.calls = min(self.calls, CONFIG.CALLS_PER_MIN)
                self.tokens = min(self.tokens, CONFIG.TOKENS_PER_MIN)
                self.last_update = now
                if self.calls >= 1 and self.tokens >= token_cost:
                    self.calls -= 1
                    self.tokens -= token_cost
                    return
            with self.lock:
                sleep_calls = max(0, (1 - self.calls) / (CONFIG.CALLS_PER_MIN / 60.0))
                sleep_tokens = max(0, (token_cost - self.tokens) / (CONFIG.TOKENS_PER_MIN / 60.0))
            sleep_sec = max(sleep_calls, sleep_tokens, 0.1)
            logging.debug(f"Rate limit: sleeping {sleep_sec:.2f}s")
            time.sleep(sleep_sec)
limiter = FunnelRateLimiter()
# ----------------------------------------------------------------------
# PROMPT BUILDER (per criterion + excluded handles)
# ----------------------------------------------------------------------
def build_prompt(criterion_id: int, excluded_handles: Set[str]) -> str:
    c = CONFIG.CRITERIA[criterion_id]
    if "queries" in c:
        selected_query = random.choice(c["queries"])
        query = f"{selected_query} since:{c.get('since', '2020-01-01')}"
    elif "query_groups" in c:
        group_strs = []
        for group in c["query_groups"]:
            if len(group) <= 1:
                selected = group
            else:
                k = max(1, round(0.8 * len(group)))
                selected = random.sample(group, k)
            group_str = '(' + ' OR '.join(f'"{term}"' for term in selected) + ')'
            group_strs.append(group_str)
        query = ' AND '.join(group_strs) + f" since:{c.get('since', '2020-01-01')}"
    else:
        query = f"{c['query']} since:{c.get('since', '2020-01-01')}"
    search_block = f"Call x_semantic_search:\n{query}"
    if c["search_type"] == "keyword":
        search_block = f"Call x_keyword_search:\n{query}"
    excluded_str = ""
    if excluded_handles:
        excluded_str = f"\n**EXCLUDE THESE HANDLES ENTIRELY** (already found in prior runs):\n" + ", ".join(excluded_handles)
    filters = f"""
For every candidate:
1. Call `x_user_search` once with query=handle (without @).
2. **Only keep if ALL pass**:
   - created_at > {CONFIG.MIN_AGE_YEARS} year ago
   - {CONFIG.MIN_FOLLOWERS} ≤ followers_count ≤ {CONFIG.MAX_FOLLOWERS}
   - following_count ≤ followers_count * {CONFIG.MAX_FOLLOWING_RATIO}
   - tweet_count ≥ {CONFIG.MIN_TWEETS}
   - profile_image_url NOT contains "{CONFIG.DEFAULT_PROFILE_IMAGE_CONTAINS}"
   - description length ≥ {CONFIG.MIN_BIO_LENGTH}
   - verified_type NOT "{CONFIG.FORBIDDEN_VERIFIED_TYPE}"
   - name + description NOT contains: {', '.join(CONFIG.BLOCKED_KEYWORDS)}
3. **Never guess.** Use only real search results.
{excluded_str}
"""
    prompt = f"""
You are Grok with access to: x_keyword_search, x_semantic_search, x_user_search.
TASK: For **Criterion {criterion_id} ({c['name']})**, find ** around {CONFIG.HANDLES_PER_RUN} new unique handles** (not previously found).
{filters}
SEARCH:
{search_block}
INSTRUCTION:
{c['instruction']}
OUTPUT **only** valid JSON:
```json
{{
  "handles": [
    {{"handle": "@user", "reason": "1 short sentence", "criteria_id": {criterion_id}}}
  ]
}}
```"""
    return prompt.strip()
# ----------------------------------------------------------------------
# API CALL
# ----------------------------------------------------------------------
def call_grok(prompt: str) -> Dict:
    logging.debug(f"\n=== API CALL START ===\nPrompt:\n{prompt}\n=== END PROMPT ===\n")
    limiter.wait()
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": CONFIG.MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 10_000
    }
    try:
        resp = requests.post(CONFIG.API_URL, headers=headers, json=payload, timeout=90)
        resp.raise_for_status()
        response_json = resp.json()
        logging.debug(f"\n=== API RESPONSE ===\n{json.dumps(response_json, indent=2)}\n=== END RESPONSE ===\n")
        return response_json
    except Exception as e:
        logging.error(f"API error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(e.response.text)
            logging.debug(f"\n=== API ERROR RESPONSE ===\n{e.response.text}\n=== END ERROR ===\n")
        return None
# ----------------------------------------------------------------------
# PARSE JSON FROM RESPONSE
# ----------------------------------------------------------------------
def extract_json(content: str) -> List[Dict]:
    match = re.search(r"```json\s*([\s\S]*?)\s*```", content, re.IGNORECASE)
    if not match:
        return []
    try:
        return json.loads(match.group(1)).get("handles", [])
    except json.JSONDecodeError as e:
        logging.error(f"JSON parse error: {e}")
        return []
# ----------------------------------------------------------------------
# VERIFY HANDLES (UPDATED WITH RETRIES AND STRICTER HANDLING)
# ----------------------------------------------------------------------
def verify_handles(all_handles: List[Dict]) -> List[Dict]:
    if not all_handles:
        return []
    valid_entries = []
    retries = 3
    delay = 1.5 # seconds between retries
    print("Starting parallel handle verification process...", flush=True)
    print(f"Total handles to verify: {len(all_handles)}", flush=True)

    def verify_single(entry):
        raw_handle = entry["handle"].lstrip("@")
        handle = raw_handle.lower()
        is_valid = False
        print(f"Verifying handle: @{raw_handle}", flush=True)
        for attempt in range(1, retries + 1):
            prompt = f"""
is the username for this account ("{raw_handle}") an actual X account?

how many followers does this account have? 

format of your answer should look like the following:  (yes or no), (follower count)

examples:

yes, 520323
no, 0
yes, 0

""".strip()
            logging.debug(f"Verification prompt for @{raw_handle}:\n{prompt}")
            print(f" @{raw_handle} Attempt {attempt}/{retries}: Sending prompt to Grok...", flush=True)
            try:
                resp = call_grok(prompt)
                if not resp or "choices" not in resp:
                    logging.warning(f"Handle @{raw_handle}: API error (attempt {attempt}/{retries})")
                    print(f" @{raw_handle} Attempt {attempt}: API response invalid or empty. Retrying...", flush=True)
                    time.sleep(delay)
                    continue
                content = resp["choices"][0]["message"]["content"].strip().lower()
                print(f" @{raw_handle} Attempt {attempt}: Received response: '{content}'", flush=True)
                # Parse response: "yes, <number>" or "no" or "no, <number>"
                parts = [p.strip() for p in content.split(',')]
                if parts[0] == "no":
                    is_valid = False
                    print(f" @{raw_handle} Attempt {attempt}: Validated as NO.", flush=True)
                    break
                elif parts[0] == "yes" and len(parts) == 2:
                    try:
                        num = int(parts[1])
                        if num > 100:
                            is_valid = True
                            print(f" @{raw_handle} Attempt {attempt}: Validated as YES (number: {num}).", flush=True)
                        else:
                            is_valid = False
                            print(f" @{raw_handle} Attempt {attempt}: Validated as NO (number <= 100: {num}).", flush=True)
                        break
                    except ValueError:
                        pass  # Fall through to invalid response
                # If not matched, invalid
                logging.warning(f"Handle @{raw_handle}: Invalid response '{content}' (attempt {attempt}/{retries})")
                print(f" @{raw_handle} Attempt {attempt}: Invalid response (not yes,<number> or no). Retrying...", flush=True)
                time.sleep(delay)
            except Exception as e:
                logging.error(f"Handle @{raw_handle}: Exception on attempt {attempt}: {e}")
                print(f" @{raw_handle} Attempt {attempt}: Exception occurred: {e}. Retrying...", flush=True)
                time.sleep(delay)
        if is_valid:
            logging.debug(f"✓ @{raw_handle} is valid")
            print(f"Handle @{raw_handle} is VALID. Adding to results.", flush=True)
            return entry
        else:
            logging.debug(f"✗ @{raw_handle} is invalid or failed verification")
            print(f"Handle @{raw_handle} is INVALID or failed. Discarding.", flush=True)
            return None

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(verify_single, entry) for entry in all_handles]
        for future in as_completed(futures):
            result = future.result()
            if result:
                valid_entries.append(result)
            print(f"Progress: {len(valid_entries)}/{len(all_handles)} valid", flush=True)

    logging.info(f"Verification complete: {len(all_handles)} → {len(valid_entries)} valid handles")
    print("\nVerification process complete.", flush=True)
    print(f"Valid handles found: {len(valid_entries)}", flush=True)
    return valid_entries
# ----------------------------------------------------------------------
# MAIN: 9 criteria × 5 runs
# ----------------------------------------------------------------------
def funnel_main():
    all_results = []
    global_excluded: Set[str] = set()
    total_expected = len(CONFIG.CRITERIA) * CONFIG.HANDLES_PER_RUN
    logging.info(f"Starting high-quality funnel: {CONFIG.RUNS_PER_CRITERION} runs × {len(CONFIG.CRITERIA)} criteria")
    for criterion_id in CONFIG.CRITERIA.keys():
        logging.info(f"── Criterion {criterion_id}: {CONFIG.CRITERIA[criterion_id]['name']} ──")
        found_in_criterion = 0
        for run in range(1, CONFIG.RUNS_PER_CRITERION + 1):
            prompt = build_prompt(criterion_id, global_excluded)
            if CONFIG.DEBUG_MODE:
                print(f"\n--- RUN {run} / CRITERION {criterion_id} ---\n{prompt}\n")
            resp = call_grok(prompt)
            if not resp or "choices" not in resp:
                logging.warning(f"Run {run} failed. Retrying in next cycle...")
                time.sleep(2)
                continue
            content = resp["choices"][0]["message"]["content"]
            handles = extract_json(content)
            new_handles = []
            for h in handles:
                handle = h["handle"].lstrip("@").lower()
                if handle not in global_excluded:
                    new_handles.append(h)
                    global_excluded.add(handle)
            all_results.extend(new_handles)
            found_in_criterion += len(new_handles)
            logging.info(f" Run {run}: +{len(new_handles)} new → {found_in_criterion}/{CONFIG.HANDLES_PER_RUN * CONFIG.RUNS_PER_CRITERION} total")
            if len(new_handles) == 0:
                logging.info(" No new handles. Moving to next criterion.")
                break
            time.sleep(CONFIG.DELAY_BETWEEN_CALLS) # Be gentle
        logging.info(f"Criterion {criterion_id} complete: {found_in_criterion} unique handles")
    # ------------------------------------------------------------------
    # VERIFY
    # ------------------------------------------------------------------
    all_results = verify_handles(all_results)
    # ------------------------------------------------------------------
    # SAVE & DISPLAY
    # ------------------------------------------------------------------
    final_output = {"handles": all_results}
    with open(CONFIG.OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    logging.info(f"Results saved to {CONFIG.OUTPUT_JSON_FILE}")
    if CONFIG.OUTPUT_TABLE and all_results:
        table = [[h.get("handle", ""), h.get("reason", ""), f"C{h.get('criteria_id', '?')}"] for h in all_results]
        print("\n" + tabulate(table, headers=["Handle", "Reason", "Crit"], tablefmt="grid"))
        print(f"\nTotal unique high-quality candidates: {len(all_results)}\n")
        print(f"DONEZO")

if __name__ == "__main__":
    funnel_main()