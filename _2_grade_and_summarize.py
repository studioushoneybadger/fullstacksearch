# File: _2_grade_and_summarize.py

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

GROK_API_KEY = os.environ['GROK_API_KEY']

"""
Grok API X Engineer Evaluation
Evaluates specific X accounts from reply analysis against 9 full-stack engineer criteria.
"""
# ============================================================================
# CONFIGURATION - Tweak these parameters as needed
# ============================================================================
# Debugging Configuration
DEBUG = False  # Set to True to enable printing of prompts and API responses

# Grok API Configuration
GROK_API_URL = 'https://api.x.ai/v1/chat/completions' # Grok API endpoint
GROK_MODEL = 'grok-4-fast-reasoning' # Model to use for searches
# Search Configuration
SEARCH_LIMIT_PER_QUERY = 50 # Number of results to retrieve per search query
DATE_RANGE_DAYS = 360 # Search within the last N days
# Parallelization Configuration
MAX_CONCURRENT_REQUESTS = 10 # Number of parallel API calls (200 calls/min = ~3.3/sec, so 5 is safe)
RATE_LIMIT_CALLS_PER_MINUTE = 160 # 80% of premium+ limit: 200 calls per minute
RATE_LIMIT_TOKENS_PER_MINUTE = 250000 # 50% of premium+ limit: 500k tokens per minute
TOKEN_ESTIMATE_PER_CALL = 45000 # Conservative estimate for tokens per call (prompt + max completion), increased due to consolidation
# Scoring Configuration
SCORING_WEIGHTS = {
    'match_count': 0.8, # Weight for number of matching criteria
    'engagement': 0.1, # Weight for engagement metrics
    'recency': 0.1 # Weight for recency of posts
}
LIKE_WEIGHT = 1.0 # Weight for likes in engagement calculation
LARGE_REPLY_WEIGHT = 3.0 # Weight for replies from large accounts
LARGE_ACCOUNT_FOLLOWERS = 10000 # Minimum followers to be considered "large account"
# Input Configuration
# ============================================================================
# CONFIGURATION - Tweak these parameters as needed
# ============================================================================
# Input Configuration
REPLY_TARGETS_FILE = 'step1_funneled_candidates.json' # Updated to point to results.json
# ... (rest of the configuration remains unchanged)
# ============================================================================
# INPUT LOADING FUNCTION
# ============================================================================
def load_users_from_reply_analysis() -> List[str]:
    """
    Load usernames from the input JSON file.
    Supports formats like { "handles": [ { "handle": "@user" }, ... ] } or plain list.
    Returns a list of usernames **with** the leading '@'.
    """
    try:
        with open(REPLY_TARGETS_FILE, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        usernames: List[str] = []
        # Handle dict with 'handles' or 'users' key
        if isinstance(raw, dict):
            users_list = raw.get('handles', []) or raw.get('users', [])
            if not isinstance(users_list, list):
                print(f"Error: Expected 'handles' or 'users' to be a list in {REPLY_TARGETS_FILE}")
                return []
            for entry in users_list:
                handle = entry.get('handle') or entry.get('username') or entry.get('user') or entry.get('screen_name') if isinstance(entry, dict) else None
                if handle and isinstance(handle, str):
                    usernames.append(handle if handle.startswith('@') else f'@{handle}')
                else:
                    print(f"Warning: Skipping malformed entry in {REPLY_TARGETS_FILE}: {entry}")
        # Handle plain list
        elif isinstance(raw, list):
            for entry in raw:
                handle = entry.get('handle') or entry.get('username') or entry.get('user') or entry.get('screen_name') if isinstance(entry, dict) else None
                if handle and isinstance(handle, str):
                    usernames.append(handle if handle.startswith('@') else f'@{handle}')
                else:
                    print(f"Warning: Skipping malformed entry in {REPLY_TARGETS_FILE}: {entry}")
        else:
            print(f"Error: {REPLY_TARGETS_FILE} must contain either a list or an object with a 'handles'/'users' key")
            return []
        # Remove duplicates while preserving order
        seen = set()
        uniq: List[str] = []
        for u in usernames:
            if u not in seen:
                seen.add(u)
                uniq.append(u)
        print(f"Loaded {len(uniq)} unique username(s) from {REPLY_TARGETS_FILE}")
        return uniq
    except FileNotFoundError:
        print(f"Error: {REPLY_TARGETS_FILE} not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse {REPLY_TARGETS_FILE}: {e}")
        return []
    except Exception as e:
        print(f"Error: Unexpected issue loading {REPLY_TARGETS_FILE}: {e}")
        return []
# Output Configuration
TOP_N_RESULTS = 15 # Number of top engineers to output
OUTPUT_JSON_FILE = 'step2_raw_grades.json' # JSON output filename
# Retry Configuration
MAX_RETRIES = 5 # Maximum number of retries for failed requests
RETRY_BACKOFF_BASE = 2 # Base multiplier for exponential backoff
# ============================================================================
# CRITERIA DEFINITIONS
# ============================================================================
CRITERIA_DEFINITIONS = {
    'End-to-End Ownership': {
        'keywords': ['end-to-end ownership', '0-to-1', 'ship solo', 'full-stack ownership', 'research to production', 'database to UI', 'deploy monitor', 'own features at scale'],
        'description': 'Ship solo from research to DB, API, UI, deploy, monitor; own 0-to-1 features at scale'
    },
    'Modern Stack Mastery': {
        'keywords': ['TypeScript', 'Python', 'Rust', 'React', 'Next.js', 'Vue', 'FastAPI', 'Node.js', 'Express', 'Docker', 'AWS', 'GCP', 'Azure', 'cloud infrastructure'],
        'description': 'TypeScript + Python/Rust; React/Next.js + FastAPI/Node; Docker + cloud (AWS/GCP)'
    },
    'TypeScript Everywhere': {
        'keywords': ['full-stack TypeScript', 'TypeScript backend', 'tRPC', 'Nest.js', 'zero runtime bugs', 'type safety', 'end-to-end types'],
        'description': 'Fullstack .ts (frontend + backend via tRPC/Nest); zero runtime bugs'
    },
    'Performance & System Design': {
        'keywords': ['low-latency', 'p95', 'p99', '<100ms', 'WebSocket', 'WebRTC', 'real-time', 'scalable pipelines', 'caching', 'rate limiting', 'system design'],
        'description': 'Low-latency (WebSocket/WebRTC), scalable pipelines, <100ms p95, caching, rate limiting'
    },
    'DevOps & Reliability': {
        'keywords': ['CI/CD', 'GitHub Actions', 'GitLab CI', 'Jenkins', 'observability', 'Sentry', 'Datadog', 'monitoring', 'Terraform', 'Kubernetes', 'K8s', 'infrastructure as code'],
        'description': 'CI/CD (GitHub Actions), observability (Sentry/Datadog), infra basics (Terraform/K8s)'
    },
    'Product Thinking': {
        'keywords': ['Figma', 'design collaboration', 'product specs', 'MVP', 'minimum viable product', 'ship fast', 'user validation', 'user feedback', 'product thinking'],
        'description': 'Talk Figma, challenge specs, ship MVP in <48h, validate with real users'
    },
    'Speed & Debugging': {
        'keywords': ['performance optimization', 'milliseconds', 'bytes optimization', 'debugging', 'production outage', 'bisect', 'debug in <30min', 'velocity', 'ship fast', 'debugging skills'],
        'description': 'Obsess over ms/bytes; bisect prod outages in <30 min; bias for velocity'
    },
    'AI/LLM Integration': {
        'keywords': ['Grok', 'LLM integration', 'AI features', 'prompt chaining', 'RAG', 'retrieval augmented generation', 'function calling', 'streaming UX', 'AI UX'],
        'description': 'Deep Grok/LLM hooks: prompt chaining, RAG, function calling, streaming UX'
    },
    'Domain Research & Pipelines': {
        'keywords': ['high-throughput', 'media systems', 'data pipelines', 'WebRTC', 'audio encoding', 'video processing', 'vector database', 'vector DB', 'embeddings', 'semantic search'],
        'description': 'Build high-throughput media/data systems (WebRTC, audio encoding, vector DBs)'
    }
}
# ============================================================================
# RATE LIMITER CLASSES
# ============================================================================
class RateLimiter:
    """
    Simple sliding window rate limiter for API calls.
    """
    def __init__(self, calls_per_period: int, period: float):
        self.calls_per_period = calls_per_period
        self.period = period
        self.timestamps: List[float] = []
        self.lock = Lock()
    def acquire(self):
        with self.lock:
            now = time.time()
            # Remove expired timestamps
            while self.timestamps and self.timestamps[0] < now - self.period:
                self.timestamps.pop(0)
            if len(self.timestamps) >= self.calls_per_period:
                wait_time = (self.timestamps[0] + self.period) - now
                if wait_time > 0:
                    time.sleep(wait_time)
                    now = time.time()
                    # Remove expired after sleep
                    while self.timestamps and self.timestamps[0] < now - self.period:
                        self.timestamps.pop(0)
            self.timestamps.append(now)
class TokenRateLimiter:
    """
    Sliding window rate limiter for tokens, with pre-call estimate and post-call actual adjustment.
    """
    def __init__(self, tokens_per_period: int, period: float, estimate_per_call: int):
        self.tokens_per_period = tokens_per_period
        self.period = period
        self.entries: List[Tuple[float, int]] = [] # (timestamp, tokens)
        self.estimate_per_call = estimate_per_call
        self.lock = Lock()
    def acquire(self):
        with self.lock:
            now = time.time()
            # Remove expired entries
            while self.entries and self.entries[0][0] < now - self.period:
                self.entries.pop(0)
            current_total = sum(tokens for _, tokens in self.entries)
            if current_total + self.estimate_per_call > self.tokens_per_period:
                wait_time = (self.entries[0][0] + self.period) - now
                if wait_time > 0:
                    time.sleep(wait_time)
                    now = time.time()
                    # Remove expired after sleep
                    while self.entries and self.entries[0][0] < now - self.period:
                        self.entries.pop(0)
    def add_actual(self, actual_tokens: int):
        with self.lock:
            now = time.time()
            self.entries.append((now, actual_tokens))
# ============================================================================
# API CALL FUNCTIONS
# ============================================================================
def evaluate_user_all_criteria(username: str, criteria_defs: Dict[str, Dict[str, Any]],
                               call_limiter: RateLimiter, token_limiter: TokenRateLimiter) -> List[Dict[str, Any]]:
    """
    Evaluate a specific user against all criteria using a single Grok API call.
  
    Args:
        username: X username (with or without @)
        criteria_defs: Dictionary of all criteria definitions
        call_limiter: Shared call rate limiter
        token_limiter: Shared token rate limiter
      
    Returns:
        List of evaluation results, one per criterion
    """
    username_clean = username.strip('@')
   
    # Build the criteria descriptions string
    criteria_str = ""
    for criterion_name, criterion_def in criteria_defs.items():
        keywords_str = ', '.join(criterion_def['keywords'][:10])
        criteria_str += f"- Criterion: {criterion_name}\n Description: {criterion_def['description']}\n Relevant keywords: {keywords_str}\n\n"
   
    prompt = f"""Analyze @{username_clean}'s X (Twitter) profile and recent posts to evaluate against the following 9 criteria.
Search their profile, bio, pinned posts, and recent posts (last {DATE_RANGE_DAYS} days) for evidence.  It is OK to give the account a 0 grade.  only give a non 0 grade if there is evidence found.
Criteria:
{criteria_str}
For each criterion, determine if there's clear evidence of a match.
Be thorough but realistic. Only mark as matching if there's clear evidence. do not fabricate any responses.
Return a JSON object with this structure:
{{
  "username": "@{username_clean}",
  "evaluations": [
    {{
      "criterion": "Criterion Name",
      "matches": true/false,
      "match_score": 0.0-1.0,
      "evidence": [
        {{
          "post_text": "Relevant post content",
          "created_at": "2025-11-01T10:00:00Z",
          "likes": 100,
          "retweets": 50,
          "replies": 25,
          "post_url": "https://x.com/..."
        }}
      ],
      "reasoning": "Brief explanation of why they match or don't match"
    }},
    ... (one for each of the 9 criteria)
  ]
}}
Ensure there are exactly 9 evaluations, one for each criterion provided.

if no evidence is found, give them a 0 score.

the scoring goes from 0-1 like this:
0: no evidence found, 
0.5: mentioned in a post
1: strong proficiency

"""
    if DEBUG:
        print(f"\nDEBUG: Prompt for {username}:\n{'=' * 80}\n{prompt}\n{'=' * 80}\n")

    headers = {
        'Authorization': f'Bearer {GROK_API_KEY}',
        'Content-Type': 'application/json'
    }
  
    payload = {
        'model': GROK_MODEL,
        'messages': [
            {'role': 'user', 'content': prompt}
        ],
        'temperature': 0.0,
        'max_tokens': 4000 * 9 # Increased to handle larger response
    }
  
    for attempt in range(MAX_RETRIES):
        try:
            # Acquire rate limits before request
            call_limiter.acquire()
            token_limiter.acquire()
           
            response = requests.post(GROK_API_URL, headers=headers, json=payload, timeout=30)
          
            # Handle rate limiting from API
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', RETRY_BACKOFF_BASE ** attempt))
                print(f"Rate limited by API. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                continue
          
            response.raise_for_status()
          
            data = response.json()
           
            # Get actual token usage
            usage = data.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = prompt_tokens + completion_tokens
            token_limiter.add_actual(total_tokens)
           
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')

            if DEBUG:
                print(f"\nDEBUG: API Response for {username}:\n{'=' * 80}\n{content}\n{'=' * 80}\n")
          
            # Extract JSON from response (may be wrapped in markdown code blocks)
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
          
            try:
                result = json.loads(content)
                if isinstance(result, dict) and 'evaluations' in result:
                    evaluations = result['evaluations']
                    if isinstance(evaluations, list) and len(evaluations) == 9:
                        for eval_item in evaluations:
                            eval_item['username'] = result['username']
                        return evaluations
                    else:
                        print(f"Warning: Expected 9 evaluations for @{username_clean}, got {len(evaluations) if isinstance(evaluations, list) else type(evaluations)}")
                else:
                    print(f"Warning: Unexpected response structure for @{username_clean}")
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON for @{username_clean}: {e}")
                print(f"Response content: {content[:200]}...")
              
            # If parsing fails, return default non-matches
            return [
                {'username': username, 'criterion': crit, 'matches': False, 'match_score': 0.0, 'evidence': [], 'reasoning': 'Parsing failed'}
                for crit in criteria_defs.keys()
            ]
              
        except requests.exceptions.RequestException as e:
            # On error, add 0 tokens (or estimate) to not block the limiter
            token_limiter.add_actual(0)
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_BACKOFF_BASE ** attempt
                print(f"Request failed for @{username_clean} (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Error evaluating @{username_clean}: {e}")
                return [
                    {'username': username, 'criterion': crit, 'matches': False, 'match_score': 0.0, 'evidence': [], 'reasoning': 'Request failed'}
                    for crit in criteria_defs.keys()
                ]
  
    return [
        {'username': username, 'criterion': crit, 'matches': False, 'match_score': 0.0, 'evidence': [], 'reasoning': 'Max retries exceeded'}
        for crit in criteria_defs.keys()
    ]
# ============================================================================
# SCORING & RANKING FUNCTIONS
# ============================================================================
def calculate_engagement_score(evidence: List[Dict[str, Any]]) -> float:
    """
    Calculate weighted engagement score from evidence posts.
  
    Args:
        evidence: List of evidence posts with engagement metrics
      
    Returns:
        Weighted engagement score
    """
    total_engagement = 0.0
  
    for post in evidence:
        likes = post.get('likes', 0)
        replies = post.get('replies', 0)
      
        # Weight likes
        total_engagement += likes * LIKE_WEIGHT
      
        # Note: Large account replies would need to be determined from reply_authors data
        # For now, we'll use replies count as a proxy
        total_engagement += replies * 0.5
  
    return total_engagement
def calculate_recency_score(created_at: str) -> float:
    """
    Calculate recency score (higher = more recent).
  
    Args:
        created_at: ISO format datetime string
      
    Returns:
        Recency score between 0 and 1
    """
    try:
        post_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        now = datetime.now(post_date.tzinfo) if post_date.tzinfo else datetime.now(timezone.utc)
        days_ago = (now - post_date.replace(tzinfo=None)).days if not post_date.tzinfo else (now - post_date).days
      
        # Score decreases with age, max at 0 days, min at DATE_RANGE_DAYS
        if days_ago < 0:
            days_ago = 0
        if days_ago > DATE_RANGE_DAYS:
            return 0.0
      
        return 1.0 - (days_ago / DATE_RANGE_DAYS)
    except (ValueError, AttributeError):
        return 0.5 # Default score if parsing fails
def score_user(user_data: Dict[str, Any], matches: List[str]) -> float:
    """
    Calculate overall score for a user.
  
    Args:
        user_data: Aggregated user data
        matches: List of criterion names that match
      
    Returns:
        Overall score
    """
    match_count = len(matches)
    match_score = min(match_count / 9.0, 1.0) # Normalize to 0-1
  
    engagement = user_data.get('total_engagement', 0)
    # Normalize engagement (assume max ~10000 for normalization)
    normalized_engagement = min(engagement / 10000.0, 1.0)
  
    recency = user_data.get('avg_recency', 0.5)
  
    score = (
        match_score * SCORING_WEIGHTS['match_count'] +
        normalized_engagement * SCORING_WEIGHTS['engagement'] +
        recency * SCORING_WEIGHTS['recency']
    )
  
    return score * 10 # Scale to 0-10 for readability
def aggregate_evaluations(evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate evaluation results by user and calculate scores.
  
    Args:
        evaluations: List of evaluation results for each user-criterion pair
      
    Returns:
        List of users with aggregated scores, sorted by score
    """
    user_data: Dict[str, Dict[str, Any]] = {}
  
    for eval_result in evaluations:
        username = eval_result.get('username', '').lower()
        if not username:
            continue
      
        if username not in user_data:
            user_data[username] = {
                'username': eval_result.get('username', ''),
                'matches': [],
                'match_scores': [],
                'all_evidence': [],
                'criterion_details': {}
            }
      
        user = user_data[username]
        criterion = eval_result.get('criterion', '')
      
        if eval_result.get('matches', False):
            user['matches'].append(criterion)
            user['match_scores'].append(eval_result.get('match_score', 0.0))
            user['all_evidence'].extend(eval_result.get('evidence', []))
            user['criterion_details'][criterion] = {
                'match_score': eval_result.get('match_score', 0.0),
                'reasoning': eval_result.get('reasoning', ''),
                'evidence_count': len(eval_result.get('evidence', []))
            }
  
    # Calculate final scores
    ranked_users = []
    for username, data in user_data.items():
        matches = sorted(data['matches'])
        match_count = len(matches)
      
        # Average match score across matched criteria
        avg_match_score = (
            sum(data['match_scores']) / len(data['match_scores'])
            if data['match_scores'] else 0.0
        )
      
        # Calculate engagement from evidence
        total_engagement = calculate_engagement_score(data['all_evidence'])
      
        # Calculate recency from evidence
        recency_scores = [
            calculate_recency_score(ev.get('created_at', ''))
            for ev in data['all_evidence']
        ]
        avg_recency = (
            sum(recency_scores) / len(recency_scores)
            if recency_scores else 0.5
        )
      
        # Calculate overall score
        user_data_for_scoring = {
            'total_engagement': total_engagement,
            'avg_recency': avg_recency
        }
        score = score_user(user_data_for_scoring, matches)
      
        ranked_users.append({
            'username': data['username'],
            'score': round(score, 2),
            'match_count': match_count,
            'matches': matches,
            'avg_match_score': round(avg_match_score, 2),
            'total_engagement': round(total_engagement, 2),
            'avg_recency': round(avg_recency, 2),
            'evidence_count': len(data['all_evidence']),
            'criterion_details': data['criterion_details']
        })
  
    # Sort by score descending
    ranked_users.sort(key=lambda x: x['score'], reverse=True)
  
    return ranked_users[:TOP_N_RESULTS] if TOP_N_RESULTS > 0 else ranked_users
# ============================================================================
# MAIN FUNCTION
# ============================================================================
def grade_main():
    """Main function to evaluate users from reply analysis against 9 criteria."""
    print("=" * 80)
    print("Grok API X Engineer Evaluation")
    print("=" * 80)
    print()
  
    # Load users from reply analysis
    print(f"Loading users from {REPLY_TARGETS_FILE}...")
    usernames = load_users_from_reply_analysis()
  
    if not usernames:
        print("No users found. Exiting.")
        return
  
    print(f"Found {len(usernames)} users to evaluate")
    print(f"Evaluating against 9 criteria with 1 API call per user...")
    print(f"Using {MAX_CONCURRENT_REQUESTS} parallel workers\n")
  
    # Initialize rate limiters
    call_limiter = RateLimiter(RATE_LIMIT_CALLS_PER_MINUTE, 60.0)
    token_limiter = TokenRateLimiter(RATE_LIMIT_TOKENS_PER_MINUTE, 60.0, TOKEN_ESTIMATE_PER_CALL)
  
    # Create evaluation tasks per user
    tasks = [(username, CRITERIA_DEFINITIONS) for username in usernames]
  
    total_evals = len(tasks)
    completed_counter = {'value': 0}
    all_evaluations = []
    lock = Lock()
  
    def process_evaluation(username: str, criteria_defs: Dict[str, Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Process a single user evaluation across all criteria."""
        results = evaluate_user_all_criteria(username, criteria_defs, call_limiter, token_limiter)
        with lock:
            completed_counter['value'] += 1
            completed = completed_counter['value']
            match_count = sum(1 for r in results if r.get('matches', False))
            avg_score = sum(r.get('match_score', 0.0) for r in results) / len(results) if results else 0.0
            print(f" [{completed}/{total_evals}] {username}: {match_count}/9 matches (avg score: {avg_score:.2f})")
        return username, results
  
    # Process all evaluations in parallel
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_evaluation, username, criteria_defs): username
            for username, criteria_defs in tasks
        }
      
        # Collect results as they complete
        for future in as_completed(future_to_task):
            try:
                username, results = future.result()
                all_evaluations.extend(results)
            except Exception as e:
                username = future_to_task[future]
                print(f"Error evaluating {username}: {e}")
                all_evaluations.extend([
                    {'username': f'@{username.strip("@")}', 'criterion': crit, 'matches': False, 'match_score': 0.0, 'evidence': []}
                    for crit in CRITERIA_DEFINITIONS.keys()
                ])
  
    print(f"\nEvaluation complete. Aggregating results...\n")
  
    # Aggregate and rank
    ranked_engineers = aggregate_evaluations(all_evaluations)
  
    # Output to console
    print("=" * 80)
    print(f"Top {len(ranked_engineers)} Engineers (by score):")
    print("=" * 80)
  
    for rank, engineer in enumerate(ranked_engineers, 1):
        print(f"\n{rank}. {engineer['username']} - Score: {engineer['score']}")
        print(f" Matches ({engineer['match_count']}/9): {', '.join(engineer['matches'])}")
        print(f" Avg Match Score: {engineer['avg_match_score']:.2f}")
        print(f" Engagement: {engineer['total_engagement']:.2f} | "
              f"Recency: {engineer['avg_recency']:.2f} | "
              f"Evidence Posts: {engineer['evidence_count']}")
      
        # Show criterion details
        if engineer['criterion_details']:
            print(" Criterion Details:")
            for crit, details in engineer['criterion_details'].items():
                print(f" - {crit}: {details['match_score']:.2f} "
                      f"({details['evidence_count']} posts)")
  
    # Output to JSON
    output_data = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'config': {
            'input_file': REPLY_TARGETS_FILE,
            'date_range_days': DATE_RANGE_DAYS,
            'scoring_weights': SCORING_WEIGHTS,
            'like_weight': LIKE_WEIGHT,
            'large_reply_weight': LARGE_REPLY_WEIGHT
        },
        'total_users_evaluated': len(usernames),
        'ranked_engineers': ranked_engineers,
        'all_evaluations': all_evaluations
    }
  
    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
   
    # New: Summary of top 15 accounts as separate JSON
    with open('_15fullstackengineers.json', 'w', encoding='utf-8') as f:
        json.dump(ranked_engineers, f, indent=2, ensure_ascii=False)
   
    # New: Export top 15 handles to CSV
    with open('_15potential_fullstack_candidates.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['handle']) # Header
        for engineer in ranked_engineers:
            writer.writerow([engineer['username']])
  
    print(f"\n{'=' * 80}")
    print(f"Results saved to {OUTPUT_JSON_FILE}")
    print(f"Top 15 summary saved to _15fullstackengineers.json")
    print(f"Top 15 handles exported to _15potential_fullstack_candidates.csv")
    print(f"DONEZO")

if __name__ == "__main__":
    grade_main()
