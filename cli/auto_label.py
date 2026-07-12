"""
cli/auto_label.py
-----------------
Generates REAL ground-truth routing labels by:
  1. Running each prompt through the local model (Ollama / qwen2.5:0.5b)
  2. Asking Tier-1 cloud judge to grade the local output: YES or NO
  3. Saving local_correct=1 (YES) or 0 (NO) for each prompt
  4. Embedding with all-MiniLM-L6-v2
  5. Writing a new datasets/ml_router_cache.json for cli/train_router.py

This replaces the heuristic labels (difficulty/domain/prompt_length rules) with
ACTUAL local model performance - far more accurate training signal.

Usage:
    python cli/auto_label.py                     # label all 779 rows
    python cli/auto_label.py --limit 200         # quick 15-min run
    python cli/auto_label.py --resume            # resume after crash

Cost:
    ~50K tokens at 0.20/1K = ~$0.10 for all 779 prompts
"""

import sys, os, csv, json, time, argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

SWEEP_CSV    = ROOT / "data_builder" / "dataset_sweep.csv"
DATASETS_DIR = ROOT / "datasets"
CACHE_PATH   = DATASETS_DIR / "ml_router_cache.json"
CHECKPOINT   = DATASETS_DIR / "auto_label_checkpoint.json"
DATASETS_DIR.mkdir(exist_ok=True)

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "qwen2.5:0.5b")

JUDGE_TEMPLATE = (
    "You are a strict answer quality evaluator.\n\n"
    "Question: {prompt}\n\n"
    "Answer given: {response}\n\n"
    "Is this answer factually correct and complete? "
    "Reply with EXACTLY one word - YES or NO. No explanation."
)


def run_local(prompt: str, max_tokens: int = 256):
    import requests
    chatml = (
        "<|im_start|>system\nYou are a concise, factual assistant. "
        "Answer directly in 1-3 sentences.\n<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    t0 = time.time()
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": LOCAL_MODEL, "prompt": chatml, "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.0,
                            "seed": 42, "repeat_penalty": 1.05, "top_p": 1.0},
            },
            timeout=90,
        )
        elapsed = time.time() - t0
        return (r.json().get("response", "").strip(), elapsed) if r.status_code == 200 \
            else (f"[HTTP {r.status_code}]", elapsed)
    except Exception as e:
        return f"[Ollama error: {e}]", time.time() - t0


def judge_with_cloud(prompt: str, local_response: str):
    from inference_wrapper.fireworks_client import call_tier
    judge_prompt = JUDGE_TEMPLATE.format(
        prompt=prompt.strip()[:800], response=local_response.strip()[:600]
    )
    reply, tokens, _ = call_tier("tier1", judge_prompt, max_tokens=10)
    first_word = reply.strip().split()[0].upper().strip(".,!?") if reply.strip() else ""
    return first_word == "YES", reply, tokens


def embed_prompts(prompts):
    from sentence_transformers import SentenceTransformer
    print("  [embed] Loading all-MiniLM-L6-v2...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"  [embed] Embedding {len(prompts):,} prompts...")
    vecs = embedder.encode(prompts, batch_size=64, show_progress_bar=True)
    return vecs.tolist()


def load_checkpoint():
    return json.loads(CHECKPOINT.read_text("utf-8")) if CHECKPOINT.exists() else {}

def save_checkpoint(data):
    CHECKPOINT.write_text(json.dumps(data), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=str(SWEEP_CSV))
    parser.add_argument("--limit",  type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    print(f"\n[auto_label] Loading prompts from {Path(args.input).name}...")
    prompts_meta = []
    with open(args.input, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            p = row.get("prompt", "").strip()
            if not p:
                continue
            prompts_meta.append({
                "prompt_id": row.get("prompt_id", f"row_{len(prompts_meta)}"),
                "prompt":    p,
                "domain":    row.get("domain", "general"),
                "difficulty": row.get("difficulty", "unknown"),
                "source":    row.get("source", "unknown"),
            })
            if args.limit and len(prompts_meta) >= args.limit:
                break

    print(f"  Loaded {len(prompts_meta):,} prompts.")
    checkpoint = load_checkpoint() if args.resume else {}
    if checkpoint:
        done = sum(1 for v in checkpoint.values() if "local_correct" in v)
        print(f"  Resuming: {done}/{len(prompts_meta)} already labeled.")

    print(f"\n[auto_label] Running {LOCAL_MODEL} + Tier-1 judge...")
    print(f"  Estimated time: ~{max(1, len(prompts_meta)*5//60)} minutes on CPU\n")

    labeled, total_tokens, correct, errors = [], 0, 0, 0

    for i, meta in enumerate(prompts_meta):
        pid = meta["prompt_id"]
        prompt = meta["prompt"]

        if pid in checkpoint and "local_correct" in checkpoint[pid]:
            entry = checkpoint[pid]
            labeled.append({**meta, **entry})
            if entry["local_correct"] == 1:
                correct += 1
            continue

        local_resp, local_lat = run_local(prompt)

        is_hard_fail = local_resp.startswith("[") or len(local_resp.split()) < 3
        if is_hard_fail:
            local_correct, judge_reply, judge_tokens = 0, "SKIPPED (local fail)", 0
        else:
            try:
                ok, judge_reply, judge_tokens = judge_with_cloud(prompt, local_resp)
                local_correct = int(ok)
                total_tokens += judge_tokens
            except Exception as ex:
                local_correct, judge_reply, judge_tokens = 0, f"[err: {ex}]", 0
                errors += 1

        if local_correct:
            correct += 1

        entry = {
            "local_response": local_resp[:300],
            "local_correct":  local_correct,
            "judge_reply":    judge_reply[:60],
            "local_latency":  round(local_lat, 2),
        }
        checkpoint[pid] = entry
        labeled.append({**meta, **entry})

        status = "YES" if local_correct else "NO "
        print(
            f"  [{i+1:3d}/{len(prompts_meta)}] {status} | "
            f"{local_lat:.1f}s | judge: {repr(judge_reply[:25])} | {pid[:35]}"
        )

        if (i + 1) % 10 == 0:
            save_checkpoint(checkpoint)

    save_checkpoint(checkpoint)

    # ---- print summary -------------------------------------------------------
    n = len(labeled)
    print(f"\n{'='*60}")
    print(f" LABELING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total       : {n:,}")
    print(f"  Local OK    : {correct:,}  ({correct/max(n,1):.1%})  -> label=1")
    print(f"  Cloud needed: {n-correct:,}  -> label=0")
    print(f"  Errors      : {errors}")
    print(f"  Judge tokens: {total_tokens:,}  (~${total_tokens/1000*0.20:.3f})")

    # ---- embed + save cache --------------------------------------------------
    print(f"\n[auto_label] Embedding prompts for semantic router training...")
    all_prompts = [m["prompt"] for m in labeled]
    all_labels  = [m["local_correct"] for m in labeled]
    embeddings  = embed_prompts(all_prompts)

    cache = []
    for emb, lbl, meta in zip(embeddings, all_labels, labeled):
        cache.append({
            "prompt":     meta["prompt"][:200],
            "model":      LOCAL_MODEL,
            "score":      0.90 if lbl else 0.20,
            "embedding":  emb,
            "label":      lbl,
            "domain":     meta["domain"],
            "difficulty": meta["difficulty"],
            "source":     meta["source"],
            "prompt_id":  meta["prompt_id"],
        })

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)

    size_kb = CACHE_PATH.stat().st_size / 1024
    print(f"\n[OK] Cache saved -> {CACHE_PATH}  ({size_kb:.0f} KB, {len(cache):,} entries)")
    print(f"\n  Label split:")
    print(f"    local_correct=1: {sum(all_labels)} ({sum(all_labels)/max(n,1):.1%})")
    print(f"    local_correct=0: {n - sum(all_labels)}")
    print(f"\n  Next step: python cli/train_router.py")


if __name__ == "__main__":
    main()
