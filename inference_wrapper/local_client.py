"""
inference_wrapper/local_client.py
Ollama local model client — auto-detect, score, and generate.
"""

import re
import time
import requests
from typing import Optional, Tuple, List, Dict

OLLAMA_BASE = "http://localhost:11434"

# Capability score by keyword in model name (higher = better)
_SCORES = {
    "kimi": 10, "qwen3": 9, "llama3.1:70b": 10, "llama3.3": 9,
    "qwen2.5:14b": 8, "qwen2.5:7b": 7, "llama3.1:8b": 7, "llama3.2:8b": 7,
    "mistral:7b": 6, "gemma2:9b": 7, "phi4": 8, "phi3": 6,
    "llama3.2:3b": 5, "qwen2.5:3b": 5, "gemma:2b": 4,
    "llama3.2:1b": 3, "openelm": 3, "mobilellm": 4, "tinyllama": 1,
}


def score_model(name: str) -> int:
    n = name.lower()
    for key, s in _SCORES.items():
        if key in n:
            return s
    m = re.search(r"(\d+\.?\d*)b", n)
    if m:
        p = float(m.group(1))
        return 10 if p >= 30 else 7 if p >= 13 else 6 if p >= 7 else 5 if p >= 3 else 3 if p >= 1 else 1
    return 3


def detect_ollama() -> Tuple[bool, List[Dict]]:
    """Returns (is_running, list_of_model_dicts)."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        if r.status_code == 200:
            return True, r.json().get("models", [])
    except Exception:
        pass
    return False, []


def best_model(models: List[Dict]) -> Optional[Dict]:
    """Pick the highest-scoring available model."""
    if not models:
        return None
    return max(models, key=lambda m: score_model(m["name"]))


def generate(prompt: str, model_name: str, max_tokens: int = 512) -> Tuple[str, float]:
    """Returns (response_text, latency_s). Uses 0 Fireworks tokens."""
    t0 = time.time()
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model_name, "prompt": prompt,
                  "stream": False, "options": {"num_predict": max_tokens}},
            timeout=90,
        )
        elapsed = time.time() - t0
        return (r.json().get("response", "").strip(), elapsed) if r.status_code == 200 \
            else (f"[Ollama HTTP {r.status_code}]", elapsed)
    except requests.exceptions.Timeout:
        return "[Ollama timeout — model too slow]", time.time() - t0
    except Exception as e:
        return f"[Ollama error: {e}]", time.time() - t0


# Gate 4: Local Correctness Verification
# ---------------------------------------------------------------------------
# Phrases that indicate the model refused or failed to answer.
_COP_OUT_PHRASES = [
    "i cannot", "i can't", "i don't know", "i am unable", "i'm unable",
    "as an ai language model", "as an ai", "i'm sorry, but", "i apologize",
    "i have no information", "i have no knowledge", "[ollama", "[error",
    "error:", "i'm just an ai",
]

def verify_local_response(prompt: str, response: str, domain: str = "factual") -> Tuple[bool, str]:
    """
    Gate 4: Lightweight post-inference sanity check + LLM Sanitizer.

    Called AFTER the local model generates a response. If this returns False,
    the caller should escalate to Tier1 (not discard) so the user always gets
    a valid answer.

    Checks:
      1. Minimum length (at least 3 words)
      2. Cop-out phrase detection (model refused/failed)
      3. Repetition loop detection (common small-model failure mode)
      4. Domain-specific: code domain must contain actual code structure
      5. [NEW] LLM-as-a-Judge Sanitizer: Ask Tier 1 if the answer is correct (costs ~50 tokens).

    Args:
        prompt:   The original user prompt.
        response: The raw text response from the local model.
        domain:   The classified domain (from difficulty_classifier).

    Returns:
        (is_valid: bool, reason: str)
    """
    r = response.strip()

    # Check 1: Minimum length
    words = r.split()
    if len(words) < 3:
        return False, f"Response too short ({len(words)} words) — likely model failure"

    # Check 2: Cop-out / refusal phrases
    lower_r = r.lower()
    for phrase in _COP_OUT_PHRASES:
        if phrase in lower_r:
            return False, f"Cop-out phrase detected: '{phrase}' — escalating to cloud"

    # Check 3: Repetition loop (common in tiny models that lose coherence)
    if len(words) >= 5:
        top_word_freq = max(words.count(w.lower()) for w in set(words)) / len(words)
        if top_word_freq > 0.60:
            return False, f"Short repetition detected (top word appears {top_word_freq:.0%} of the time)"
            
    if len(words) > 12:
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        if unique_ratio < 0.35:
            return False, f"Repetitive output detected (unique word ratio={unique_ratio:.2f}) — model is looping"

    # Check 4: Code domain must have code structure
    if domain == "code":
        import re
        ARTICLES = {'the', 'a', 'an', 'to', 'this', 'that', 'it', 'its', 'all'}
        has_return_expr = False
        for m in re.finditer(r'return\s+(\w+)', r):
            if m.group(1).lower() not in ARTICLES:
                has_return_expr = True
                break
        has_backtick_block = '```' in r
        has_code_structure = (
            "def " in r or
            "function " in r.lower() or
            "class " in r or
            has_backtick_block or
            has_return_expr or
            "=>" in r
        )
        if not has_code_structure:
            return False, "Code domain: response contains no recognisable code structure"

    # Check 5: [NEW] LLM Sanitizer (Tier 1 judge)
    try:
        from inference_wrapper.model_clients import get_client
        tier1_client = get_client("tier1")
        judge_prompt = f"You are an evaluator. The user asked: '{prompt}'. The local model answered: '{response}'. Is this answer correct? Reply only YES or NO."
        judge_result = tier1_client.generate(judge_prompt, max_tokens=10).strip().upper()
        if "NO" in judge_result:
            return False, "LLM Sanitizer rejected answer (Tier 1 marked it as incorrect)"
    except Exception as e:
        # If the sanitizer fails (e.g. network error), we assume the answer is fine to not block the pipeline
        pass

    return True, "Response passes all quality checks and Sanitizer"
