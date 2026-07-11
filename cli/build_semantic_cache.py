"""
cli/build_semantic_cache.py
----------------------------
Builds datasets/ml_router_cache.json from the real dataset_sweep.csv
so the semantic router can be retrained from scratch on real signal.

Replaces the 20-sample mock cache that was causing ~25% routing accuracy.

Label derivation strategy (zero inference cost — purely heuristic):
  - TIER1-CORRECT rows that are "easy" AND short AND in simple domains
    → label = 1  (local-viable: a tiny model CAN answer this)
  - Everything else
    → label = 0  (cloud-required: route to Tier 1 or Tier 2)

This is a principled proxy:
  - "easy" difficulty is assigned by the benchmark itself (reliable ground truth)
  - Short prompts (<= 20 words) remove multi-step / multi-hop queries
  - Domain filter removes code/math/science (sub-1B models fail these categories)

Expected output:
  - ~3,000 balanced samples (1,500 local + 1,500 cloud)
  - datasets/ml_router_cache.json in the format expected by cli/train_router.py
  - Estimated semantic router accuracy after retraining: 82–88%

Usage:
  python cli/build_semantic_cache.py
  python cli/train_router.py          # retrain semantic model on new cache
"""

import io
import json
import os
import sys
import random
from pathlib import Path

# Force UTF-8 output on Windows cp1252 terminals to avoid unicode crashes
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Paths ──────────────────────────────────────────────────────────────────────
SWEEP_CSV   = ROOT / "data_builder" / "dataset_sweep.csv"
DATASETS_DIR = ROOT.parent / "datasets"          # project-root/../datasets
CACHE_PATH  = DATASETS_DIR / "ml_router_cache.json"

# ── Label derivation config ────────────────────────────────────────────────────
# Domains where a sub-1B model can sometimes succeed
LOCAL_VIABLE_DOMAINS = {"factual", "general", "language"}

# Benchmark sources that are too hard for local models
HARD_SOURCES = {"gsm8k", "humaneval", "math_hard", "leetcode", "musique", "hotpotqa"}

# Max prompt length for local routing (words)
# Loosened to 25 to capture more local-viable samples (fixes class imbalance)
MAX_LOCAL_PROMPT_LEN = 25

# Target sample counts (balanced)
TARGET_LOCAL = 1500
TARGET_CLOUD = 1500

RANDOM_SEED = 42


def derive_local_label(row: dict) -> int:
    """
    Returns 1 (local-viable) or 0 (cloud-required).

    Local-viable if ALL of:
      1a. difficulty == "easy"   (benchmark metadata)
       OR 1b. difficulty == "medium" AND prompt_length <= 12 (very short medium)
      2.  prompt_length <= MAX_LOCAL_PROMPT_LEN
      3.  domain in LOCAL_VIABLE_DOMAINS  (removes code/math/science)
      4.  source NOT in HARD_SOURCES      (removes hard benchmarks)
      5.  has_code_block == 0
      6.  has_math_symbols == 0
    """
    difficulty   = str(row.get("difficulty", "")).strip().lower()
    domain       = str(row.get("domain", "")).strip().lower()
    source       = str(row.get("source", "")).strip().lower()
    prompt_len   = _safe_int(row.get("prompt_length", 999))
    has_code     = _safe_int(row.get("has_code_block", 1))
    has_math     = _safe_int(row.get("has_math_symbols", 1))

    # Difficulty gate: easy always qualifies; very short medium prompts also ok
    diff_ok = (difficulty == "easy") or (difficulty == "medium" and prompt_len <= 12)

    if (diff_ok
            and prompt_len <= MAX_LOCAL_PROMPT_LEN
            and domain in LOCAL_VIABLE_DOMAINS
            and source not in HARD_SOURCES
            and has_code == 0
            and has_math == 0):
        return 1   # local-viable
    return 0       # cloud-required


def _safe_int(val) -> int:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return 0


def main():
    import csv

    print(f"[build_semantic_cache] Loading {SWEEP_CSV} ...")
    if not SWEEP_CSV.exists():
        print(f"  [ERROR] File not found: {SWEEP_CSV}")
        sys.exit(1)

    # ── Load CSV ──────────────────────────────────────────────────────────────
    rows = []
    with open(SWEEP_CSV, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"  Loaded {len(rows):,} rows.")

    # ── Derive labels ─────────────────────────────────────────────────────────
    local_rows = []
    cloud_rows = []

    for row in rows:
        prompt = row.get("prompt", "").strip()
        if not prompt or len(prompt) < 5:
            continue   # skip empty/junk rows

        label = derive_local_label(row)
        if label == 1:
            local_rows.append((prompt, label))
        else:
            cloud_rows.append((prompt, label))

    print(f"  Local-viable (label=1): {len(local_rows):,}")
    print(f"  Cloud-required (label=0): {len(cloud_rows):,}")

    # ── Balance + subsample ───────────────────────────────────────────────────
    random.seed(RANDOM_SEED)
    random.shuffle(local_rows)
    random.shuffle(cloud_rows)

    sampled_local = local_rows[:TARGET_LOCAL]
    sampled_cloud = cloud_rows[:TARGET_CLOUD]

    print(f"\n  Subsampled to {len(sampled_local)} local + {len(sampled_cloud)} cloud "
          f"= {len(sampled_local) + len(sampled_cloud)} total training samples.")

    if len(sampled_local) < 50:
        print("  [WARNING] Very few local-viable samples found — check difficulty column values.")

    # ── Embed with all-MiniLM-L6-v2 ──────────────────────────────────────────
    print("\n[build_semantic_cache] Loading sentence-transformers model...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  [ERROR] sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("  Model loaded. Embedding prompts (this takes ~10–30 seconds on CPU)...")

    all_samples = sampled_local + sampled_cloud
    prompts = [s[0] for s in all_samples]
    labels  = [s[1] for s in all_samples]

    # Batch-encode (all-MiniLM-L6-v2 does ~500 sentences/sec on CPU)
    embeddings = embedder.encode(prompts, batch_size=64, show_progress_bar=True)

    print(f"  Embedded {len(prompts):,} prompts -> shape {embeddings.shape}")

    # ── Build cache entries ───────────────────────────────────────────────────
    # Format expected by cli/train_router.py:
    # [{"embedding": [float, ...], "model_scores": {"qwen2.5:0.5b": float}}, ...]
    # We use the label directly: label=1 -> score=0.9 (above THRESHOLD=0.65)
    #                             label=0 -> score=0.3 (below THRESHOLD=0.65)
    TARGET_MODEL = "qwen2.5:0.5b"
    SCORE_LOCAL  = 0.90   # clearly above the 0.65 threshold -> trains as "local success"
    SCORE_CLOUD  = 0.20   # clearly below the 0.65 threshold -> trains as "cloud fallback"

    cache = []
    for emb, label in zip(embeddings, labels):
        score = SCORE_LOCAL if label == 1 else SCORE_CLOUD
        cache.append({
            "embedding":    emb.tolist(),
            "model_scores": {TARGET_MODEL: score},
        })

    # ── Save ──────────────────────────────────────────────────────────────────
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)

    size_kb = CACHE_PATH.stat().st_size / 1024
    print(f"\n[OK] Cache saved -> {CACHE_PATH}  ({size_kb:.0f} KB, {len(cache):,} entries)")
    print(f"\n  Label distribution:")
    print(f"    Local-viable  (label=1, score={SCORE_LOCAL}): {labels.count(1)}")
    print(f"    Cloud-required(label=0, score={SCORE_CLOUD}): {labels.count(0)}")
    imbalance = labels.count(1) / max(len(labels), 1)
    print(f"    Local ratio: {imbalance:.1%}  (class_weight='balanced' in LR handles this)")
    print(f"\n  Next step: python cli/train_router.py")


if __name__ == "__main__":
    main()
