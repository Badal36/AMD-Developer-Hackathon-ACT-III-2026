"""
calibration/profile.py
-----------------------
CapabilityProfile: 2D model capability map (domain × difficulty).

Schema stored in ~/.hybridrouter/config.json under each model:

  "capability_profile": {
      "math":        {"L1": 1.0, "L2": 0.95, "L3": 0.82, "L4": 0.41},
      "factual":     {"L1": 1.0, "L2": 0.91, "L3": 0.78, "L4": 0.52},
      "code":        {"L1": 1.0, "L2": 0.80, "L3": 0.55, "L4": 0.22},
      "language":    {"L1": 1.0, "L2": 0.88, "L3": 0.70, "L4": 0.38},
      "reasoning":   {"L1": 1.0, "L2": 0.75, "L3": 0.58, "L4": 0.30},
      "instruction": {"L1": 1.0, "L2": 0.85, "L3": 0.60, "L4": 0.25}
  },
  "routing_thresholds": {
      "math":        {"max_local_level": "L3", "confidence_floor": 0.70},
      ...
  }

Routing thresholds are derived from the capability_profile automatically
using the policy: route locally if the model's measured success rate at
that (domain, level) >= ROUTING_CONFIDENCE_FLOOR.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

# ── Constants ────────────────────────────────────────────────────────────────

DOMAINS   = ["math", "factual", "code", "language", "reasoning", "instruction"]
LEVELS    = ["L1", "L2", "L3", "L4"]

# Source → domain mapping (from calibration_prompts.jsonl sources)
SOURCE_TO_DOMAIN: Dict[str, str] = {
    "mmlu":       "factual",
    "arc":        "factual",
    "gsm8k":      "math",
    "humaneval":  "code",
    "truthfulqa": "language",
    "math_hard":  "math",
    "leetcode":   "code",
    "musique":    "reasoning",
    "hotpotqa":   "reasoning",
    "ifeval":     "instruction",
}

# Routing confidence floor: model must score >= this at a (domain, level)
# to be trusted for local routing of that level.
ROUTING_CONFIDENCE_FLOOR = 0.65


# ── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class CapabilityProfile:
    """
    Stores measured accuracy per (domain, difficulty level) for one model.
    Also derives routing thresholds automatically.
    """
    model_name: str
    # acc[domain][level] = float 0..1  (None = not measured)
    acc: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure all domain/level slots exist
        for d in DOMAINS:
            self.acc.setdefault(d, {})
            for lv in LEVELS:
                self.acc[d].setdefault(lv, None)

    # ── Routing decision ────────────────────────────────────────────────────

    def max_local_level(self, domain: str) -> Optional[str]:
        """
        Highest difficulty level where the model scores >= ROUTING_CONFIDENCE_FLOOR.
        Returns None if even L1 is below floor (route all remote).
        """
        domain = domain if domain in DOMAINS else "factual"
        max_lv = None
        for lv in LEVELS:
            score = self.acc.get(domain, {}).get(lv)
            if score is not None and score >= ROUTING_CONFIDENCE_FLOOR:
                max_lv = lv
        return max_lv

    def should_route_local(self, domain: str, level: str) -> tuple[bool, str]:
        """
        Returns (route_local: bool, reason: str).
        """
        domain = domain if domain in DOMAINS else "factual"
        level  = level  if level  in LEVELS  else "L2"

        score = self.acc.get(domain, {}).get(level)
        if score is None:
            # No calibration evidence for this (domain, level) pair.
            # Returning False defers the decision to the ML router, which is
            # always safer than routing based on fabricated/bootstrapped data.
            return False, f"No calibration data for {domain}/{level} — deferring to ML router"

        max_lv = self.max_local_level(domain)
        if max_lv is None:
            return False, f"Model below confidence floor on all {domain} levels"

        level_idx    = LEVELS.index(level)
        max_lv_idx   = LEVELS.index(max_lv)

        if level_idx <= max_lv_idx:
            return True, (
                f"Model scores {score*100:.0f}% on {domain}/{level} "
                f"(>= {ROUTING_CONFIDENCE_FLOOR*100:.0f}% floor) -> LOCAL"
            )
        else:
            return False, (
                f"Model scores {score*100:.0f}% on {domain}/{level} "
                f"(< {ROUTING_CONFIDENCE_FLOOR*100:.0f}% floor) -> REMOTE"
            )

    def summary(self) -> str:
        """Human-readable one-line summary of max local levels per domain."""
        parts = []
        for d in DOMAINS:
            ml = self.max_local_level(d)
            parts.append(f"{d}:{ml or 'none'}")
        return "  ".join(parts)

    # ── Serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "capability_profile": {
                d: {lv: self.acc[d][lv] for lv in LEVELS}
                for d in DOMAINS
            },
            "routing_thresholds": {
                d: {
                    "max_local_level": self.max_local_level(d),
                    "confidence_floor": ROUTING_CONFIDENCE_FLOOR,
                }
                for d in DOMAINS
            },
        }

    @classmethod
    def from_dict(cls, model_name: str, data: dict) -> "CapabilityProfile":
        """Load from the dict stored in config.json for a model."""
        profile = cls(model_name=model_name)
        raw = data.get("capability_profile", {})
        for d in DOMAINS:
            for lv in LEVELS:
                val = raw.get(d, {}).get(lv)
                profile.acc[d][lv] = val
        return profile

    @classmethod
    def bootstrap_from_calibration_acc(
        cls,
        model_name: str,
        calibration_acc: float,
        source_stats: dict,
    ) -> "CapabilityProfile":
        """
        Create a rough profile from old flat calibration data (backward compat).
        Uses overall acc + per-source accuracy to estimate domain/level scores.
        """
        profile = cls(model_name=model_name)

        # Map source stats to domains
        domain_acc: Dict[str, float] = {}
        for src, stats in source_stats.items():
            domain = SOURCE_TO_DOMAIN.get(src, "factual")
            acc    = stats.get("acc", calibration_acc)
            # Average if multiple sources map to same domain
            if domain in domain_acc:
                domain_acc[domain] = (domain_acc[domain] + acc) / 2
            else:
                domain_acc[domain] = acc

        # Fill in domains with no source data
        for d in DOMAINS:
            if d not in domain_acc:
                domain_acc[d] = calibration_acc * 0.85  # conservative estimate

        # Assign level scores using a decay curve from the measured accuracy.
        # The idea: if overall acc is A, the model likely does:
        #   L1 ~ 1.0 (trivial, always)
        #   L2 ~ A + 0.10  (slightly above average)
        #   L3 ~ A - 0.10  (slightly below average)
        #   L4 ~ A - 0.35  (frontier tasks, significantly harder)
        for d in DOMAINS:
            base = domain_acc.get(d, calibration_acc)
            profile.acc[d]["L1"] = min(1.0, base + 0.20)
            profile.acc[d]["L2"] = min(1.0, base + 0.08)
            profile.acc[d]["L3"] = max(0.0, base - 0.12)
            profile.acc[d]["L4"] = max(0.0, base - 0.38)

        return profile


# ── Config I/O helpers ───────────────────────────────────────────────────────

CONFIG_PATH = Path.home() / ".hybridrouter" / "config.json"


def load_profile(model_name: str) -> Optional[CapabilityProfile]:
    """
    Load a CapabilityProfile from config.json.
    If only old flat data exists, bootstraps a rough profile from it.
    Returns None if model has never been calibrated.
    """
    if not CONFIG_PATH.exists():
        return None
    try:
        cfg = json.loads(CONFIG_PATH.read_text())
    except Exception:
        return None

    model_data = cfg.get("models", {}).get(model_name)
    if not model_data:
        return None

    # Modern format: has capability_profile key
    if "capability_profile" in model_data:
        return CapabilityProfile.from_dict(model_name, model_data)

    # Legacy flat format: no capability_profile available.
    # v2 policy: do NOT bootstrap fabricated scores from calibration_acc.
    # Return None so the caller falls back to the ML router (safer than invented data).
    return None


def save_profile(profile: CapabilityProfile, extra: dict = None):
    """
    Merge profile data into the model entry in config.json.
    `extra` carries additional scalar fields (calibration_acc, etc.)
    """
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        cfg = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    except Exception:
        cfg = {}

    cfg.setdefault("models", {})
    entry = cfg["models"].setdefault(profile.model_name, {})
    entry.update(profile.to_dict())
    if extra:
        entry.update(extra)

    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    # Quick sanity test
    p = CapabilityProfile("llama3.1:70b")
    p.acc["math"]["L1"] = 1.0
    p.acc["math"]["L2"] = 0.95
    p.acc["math"]["L3"] = 0.82
    p.acc["math"]["L4"] = 0.41
    p.acc["code"]["L1"] = 1.0
    p.acc["code"]["L2"] = 0.78
    p.acc["code"]["L3"] = 0.52
    p.acc["code"]["L4"] = 0.20
    print(p.summary())
    for domain, level in [("math", "L2"), ("math", "L4"), ("code", "L3")]:
        local, reason = p.should_route_local(domain, level)
        print(f"{domain}/{level}: {'LOCAL' if local else 'REMOTE'}  — {reason}")
