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


# ── ChatML system prompt for qwen2.5 / openelm sub-1B models ────────────────
# These models are trained with ChatML tokens. Without proper role wrapping,
# they default to free-form continuation which causes hallucination & loops.
_SYSTEM_PROMPT = (
    "<|im_start|>system\n"
    "You are a concise, factual assistant. Rules:\n"
    "1. Answer directly — no preamble, no filler phrases.\n"
    "2. If you are not confident, say exactly: I don't know.\n"
    "3. Keep answers under 3 sentences unless explicitly asked for more.\n"
    "4. Do NOT repeat yourself. Stop immediately after answering.\n"
    "<|im_end|>\n"
)


def _build_prompt(user_prompt: str) -> str:
    """Wrap a user prompt in ChatML format for qwen2.5/openelm models."""
    return (
        _SYSTEM_PROMPT
        + f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )


def generate(prompt: str, model_name: str, max_tokens: int = 512) -> Tuple[str, float]:
    """Returns (response_text, latency_s). Uses 0 Fireworks tokens.
    
    Enforces deterministic inference:
      - temperature=0  : greedy decoding, no sampling stochasticity
      - seed=42        : reproducible outputs on identical prompts
      - repeat_penalty : prevents looping without degrading quality
    ChatML wrapping activates the model's role-switching mechanism.
    """
    t0 = time.time()
    full_prompt = _build_prompt(prompt)
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model":  model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "num_predict":    max_tokens,
                    "temperature":    0.0,   # DETERMINISTIC — no stochastic sampling
                    "seed":           42,    # REPRODUCIBLE across identical prompts
                    "repeat_penalty": 1.05,  # LOOP PREVENTION (>1.1 degrades quality)
                    "top_p":          1.0,   # Compatible with greedy temp=0
                },
            },
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
    
    # Minimum quality score check: tiny models often output trivial content
    if len(words) < 5 and any(w.lower() in ["ok", "yes", "no", "sure"] for w in words):
        return False, "Response too trivial/short — escalating to cloud"

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

    # Check 5: LLM-as-a-Judge Sanitizer (Tier 1 judge)
    # CRITICAL DESIGN: requires explicit "YES" — any other response escalates to cloud.
    # This is intentionally strict:
    #   - If Tier 1 says "No" / "NO" / "No." → escalate (correct)
    #   - If Tier 1 response is truncated, ambiguous, or errors → escalate (safe default)
    #   - Only an explicit "YES" (case-insensitive, anywhere in the first word) passes.
    # This prevents factually wrong but well-formed local answers from reaching the user.
    try:
        from inference_wrapper.fireworks_client import call_tier
        judge_prompt = (
            f"Task: Judge if the assistant's answer is factually correct and complete.\n"
            f"Question: {prompt}\n"
            f"Answer: {response}\n"
            f"Reply with exactly one word — YES if the answer is correct, NO if it is wrong or incomplete."
        )
        judge_text, _tokens, _lat = call_tier("tier1", judge_prompt, max_tokens=10)
        # Require EXPLICIT yes — normalise to first word only
        first_word = judge_text.strip().split()[0].upper().strip(".,!?") if judge_text.strip() else ""
        if first_word != "YES":
            return False, (
                f"LLM Sanitizer: Tier 1 judge did not confirm answer "
                f"(replied: '{judge_text.strip()[:40]}') — escalating to Cloud"
            )
    except Exception as ex:
        # Sanitizer failed (network/API error) — FAIL SAFE: escalate rather than silently pass.
        # We cannot trust the local answer if we can't verify it.
        return False, f"LLM Sanitizer unavailable ({type(ex).__name__}) — escalating to Cloud as fail-safe"

    return True, "Response passes all quality checks and Sanitizer (Tier 1 confirmed YES)"
