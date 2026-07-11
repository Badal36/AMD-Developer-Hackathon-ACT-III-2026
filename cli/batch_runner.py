"""
cli/batch_runner.py
--------------------
Hackathon Judge Evaluation Runner.

Reads tasks from /input/tasks.json, routes each prompt through
the HybridRouter pipeline, and writes results to /output/results.json.

Input format (tasks.json):
  [
    {"id": "task_001", "prompt": "What is 2 + 2?"},
    {"id": "task_002", "prompt": "Write a Python function to reverse a list."},
    ...
  ]

Output format (results.json):
  [
    {
      "id": "task_001",
      "response": "4",
      "route": "local",
      "model": "qwen3:0.6b",
      "latency_ms": 120.5,
      "tokens_used": 0
    },
    ...
  ]

Usage:
  python cli/batch_runner.py --input /input/tasks.json --output /output/results.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from inference_wrapper.feature_extractor import extract_features
from inference_wrapper.router_core       import predict
from inference_wrapper.simplicity_gate   import is_trivially_simple
from inference_wrapper.local_client      import (
    detect_ollama, generate as local_gen, verify_local_response
)
from inference_wrapper.fireworks_client  import call_tier
from calibration.profile                 import load_profile
from cli.main                            import load_config, select_active_model, _composite_score


def route_one(prompt: str, active_model: str | None, cfg: dict) -> dict:
    """Route a single prompt and return a result dict."""
    t_start = time.time()

    feats = extract_features(prompt)
    capability_profile = load_profile(active_model) if active_model else None

    model_data = cfg.get("models", {}).get(active_model, {}) if active_model else {}
    model_acc  = model_data.get("calibration_acc", 0.0)
    src_stats  = model_data.get("source_stats", {})
    has_local  = bool(active_model and model_data)

    is_simple, gate_reason, gate_conf = is_trivially_simple(
        prompt, feats, model_acc,
        src_stats if has_local else None,
        capability_profile=capability_profile if has_local else None,
    )

    dest        = "tier2"
    tokens_used = 0
    response    = ""
    model_used  = "remote"

    if is_simple and has_local:
        # Try local first
        raw_response, latency = local_gen(prompt, active_model)

        # Gate 4: correctness check
        _domain = "factual"
        if capability_profile is not None:
            from inference_wrapper.difficulty_classifier import classify
            _domain = classify(prompt, feats).domain

        is_valid, _ = verify_local_response(raw_response, domain=_domain)

        if is_valid:
            response   = raw_response
            dest       = "local"
            model_used = active_model
        else:
            # Escalate to Tier1
            response, tokens_used, latency = call_tier("tier1", prompt)
            dest       = "tier1"
            model_used = "gpt-oss-20b (tier1)"
    else:
        # ML router for tier1 vs tier2
        tier, t1p, t2p = predict(feats)
        if tier == "tier1":
            response, tokens_used, latency = call_tier("tier1", prompt)
            dest       = "tier1"
            model_used = "gpt-oss-20b (tier1)"
        else:
            response, tokens_used, latency = call_tier("tier2", prompt)
            dest       = "tier2"
            model_used = "glm-5p2 (tier2)"

    total_ms = (time.time() - t_start) * 1000

    return {
        "route":      dest,
        "model":      model_used,
        "response":   response,
        "latency_ms": round(total_ms, 1),
        "tokens_used": tokens_used,
    }


def main():
    parser = argparse.ArgumentParser(description="HybridRouter batch judge runner")
    parser.add_argument("--input",  default="/input/tasks.json",   help="Path to tasks JSON")
    parser.add_argument("--output", default="/output/results.json", help="Path to results JSON")
    args = parser.parse_args()

    # Load config + select best local model
    cfg = load_config()
    running, _ = detect_ollama()
    active_model = select_active_model(cfg, interactive=False) if running else None

    if active_model:
        print(f"[batch_runner] Active local model: {active_model}")
    else:
        print("[batch_runner] No local model — remote-only mode")

    # Read tasks
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[batch_runner] ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    tasks = json.loads(input_path.read_text())
    print(f"[batch_runner] Loaded {len(tasks)} tasks from {input_path}")

    # Process each task
    results = []
    for i, task in enumerate(tasks):
        task_id = task.get("id", f"task_{i:04d}")
        prompt  = task.get("prompt", "")

        print(f"[{i+1}/{len(tasks)}] {task_id}: {prompt[:60]}...", end="", flush=True)
        try:
            result = route_one(prompt, active_model, cfg)
            result["id"] = task_id
            results.append(result)
            print(f" -> {result['route']} ({result['latency_ms']:.0f}ms)")
        except Exception as e:
            print(f" -> ERROR: {e}")
            results.append({
                "id":         task_id,
                "route":      "error",
                "model":      "none",
                "response":   f"[Router error: {e}]",
                "latency_ms": 0.0,
                "tokens_used": 0,
            })

    # Write results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n[batch_runner] Results written to {output_path}")

    # Summary
    routes = [r["route"] for r in results]
    print(f"[batch_runner] Routing summary:")
    for dest in ["local", "tier1", "tier2", "error"]:
        count = routes.count(dest)
        if count:
            print(f"  {dest:8s}: {count} ({count/len(routes)*100:.1f}%)")


if __name__ == "__main__":
    main()
