"""
inference_wrapper/fireworks_client.py
Fireworks AI remote model client — Tier 1 (gpt-oss-20b) and Tier 2 (glm-5p2).
"""

import os, time
from typing import Tuple
from openai import OpenAI

TIER1_MODEL = os.environ.get("TIER1_MODEL", "accounts/fireworks/models/gpt-oss-20b")
TIER2_MODEL = os.environ.get("TIER2_MODEL", "accounts/fireworks/models/glm-5p2")

TIER_DISPLAY = {
    "tier1": "gpt-oss-20b  [cheap]",
    "tier2": "glm-5p2      [powerful]",
}

# Approximate cost per 1K tokens (relative, for savings display)
TIER_COST_PER_1K = {"tier1": 0.20, "tier2": 0.90}
LOCAL_COST_PER_1K = 0.0


def _client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url=os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"),
    )


def call_tier(tier: str, prompt: str, max_tokens: int = 1024) -> Tuple[str, int, float]:
    """
    Call a Fireworks tier.
    Returns (response_text, total_tokens_used, latency_seconds).
    """
    model = TIER1_MODEL if tier == "tier1" else TIER2_MODEL
    t0 = time.time()
    try:
        resp = _client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        elapsed = time.time() - t0
        text    = (resp.choices[0].message.content or "").strip()
        tokens  = resp.usage.total_tokens if resp.usage else 0
        return text, tokens, elapsed
    except Exception as e:
        return f"[Fireworks error: {e}]", 0, time.time() - t0


def estimated_cost(tokens: int, tier: str) -> float:
    return tokens / 1000 * TIER_COST_PER_1K.get(tier, 0.5)


def baseline_cost(tokens: int) -> float:
    """Cost if we had always used tier2 (expensive baseline)."""
    return tokens / 1000 * TIER_COST_PER_1K["tier2"]
