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
    "llama3.2:1b": 3, "openelm": 2, "mobilellm": 2, "tinyllama": 1,
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
