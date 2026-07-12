import json
import time
import os
import sys
import argparse
import random
import csv
import numpy as np
from rich.console import Console
from rich.table import Table

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_wrapper.semantic_router import predict_local_viability
from inference_wrapper.router_core import predict as tabular_predict
from inference_wrapper.feature_extractor import extract_features
from inference_wrapper.simplicity_gate import is_trivially_simple
from inference_wrapper.local_client import generate, verify_local_response
from sentence_transformers import SentenceTransformer, util

CONSOLE = Console()

def run_accuracy_benchmark(limit=100, seed=42):
    random.seed(seed)
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_builder", "dataset_sweep.csv")
    
    if not os.path.exists(dataset_path):
        CONSOLE.print(f"[red]Error: Dataset not found at {dataset_path}[/red]")
        return
        
    all_data = []
    with open(dataset_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_data.append(row)
            
    # Sample data
    if limit < len(all_data):
        sampled_data = random.sample(all_data, limit)
    else:
        sampled_data = all_data

    CONSOLE.print(f"[bold cyan]Loaded {len(sampled_data)} prompts from dataset_sweep.csv for full accuracy benchmarking.[/bold cyan]")
    CONSOLE.print("[bold cyan]Loading SentenceTransformer (all-MiniLM-L6-v2) for local grading...[/bold cyan]")
    
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        CONSOLE.print(f"[red]Failed to load sentence-transformers: {e}[/red]")
        return

    stats = {
        "local_routed": 0,
        "tier1_routed": 0,
        "tier2_routed": 0,
        "local_correct": 0,
        "tier1_correct": 0,
        "tier2_correct": 0,
        "system_tokens": 0,
        "baseline_tier1_tokens": 0,
        "baseline_tier2_tokens": 0,
        "gate4_fails": 0,
        "gate4_escalate_tier1": 0,
        "gate4_escalate_tier2": 0,
    }

    CONSOLE.print(f"[bold magenta]Starting Benchmark (limit={limit})...[/bold magenta]")
    
    local_model = "qwen2.5:0.5b"

    for i, data in enumerate(sampled_data):
        prompt = data["prompt"]
        truth_text = data["reference_answer"]
        
        # Read pre-computed CSV stats
        t1_corr = int(data.get("tier1_correct", 0))
        t1_toks = int(data.get("tier1_tokens", 0))
        t2_corr = int(data.get("tier2_correct", 0))
        t2_toks = int(data.get("tier2_tokens", 0))
        
        stats["baseline_tier1_tokens"] += t1_toks
        stats["baseline_tier2_tokens"] += t2_toks
        
        # FULL ROUTING PIPELINE (Gate 0 -> Semantic -> Tabular)
        feats = extract_features(prompt)
        h_simple, h_reason, h_conf = is_trivially_simple(prompt, feats, 0.34, None)
        s_viable, s_reason, s_conf = predict_local_viability(prompt)
        
        # Mimic main.py logic exactly
        is_local = False
        if not h_simple and h_conf < 0.3:
            is_local = False
        elif s_viable:
            if not h_simple and h_conf >= 0.15:
                is_local = False
            else:
                is_local = True
        else:
            is_local = False
            
        if is_local:
            # --- Local path ---
            stats["local_routed"] += 1
            raw_response, lat = generate(prompt, local_model, max_tokens=150)
            
            # Gate 4: fast heuristic check for garbage
            is_valid = (
                raw_response
                and not raw_response.startswith("[")
                and len(raw_response.split()) >= 3
            )

            if not is_valid:
                # Gate 4 FAIL -> escalate to cloud (Tabular Router decides Tier1 vs Tier2)
                stats["gate4_fails"] += 1
                tier, t1p, t2p = tabular_predict(feats)
                if tier == "tier1":
                    stats["gate4_escalate_tier1"] += 1
                    stats["system_tokens"] += t1_toks
                    if t1_corr: stats["tier1_correct"] += 1
                else:
                    stats["gate4_escalate_tier2"] += 1
                    stats["system_tokens"] += t2_toks
                    if t2_corr: stats["tier2_correct"] += 1
            else:
                # Local generation valid -> grade it
                truth_emb = embedder.encode(truth_text, convert_to_tensor=True)
                resp_emb = embedder.encode(raw_response, convert_to_tensor=True)
                score = util.cos_sim(truth_emb, resp_emb)[0][0].item()
                if score > 0.65:
                    stats["local_correct"] += 1
                    
        else:
            # --- Cloud path ---
            tier, t1p, t2p = tabular_predict(feats)
            if tier == "tier1":
                stats["tier1_routed"] += 1
                stats["system_tokens"] += t1_toks
                if t1_corr: stats["tier1_correct"] += 1
            else:
                stats["tier2_routed"] += 1
                stats["system_tokens"] += t2_toks
                if t2_corr: stats["tier2_correct"] += 1

        if (i + 1) % 10 == 0:
            CONSOLE.print(f"  Processed {i + 1}/{len(sampled_data)}...")

    # --- Output Results ---
    CONSOLE.print("\n[bold green]Accuracy Benchmark Complete![/bold green]")
    
    total = len(sampled_data)
    
    total_local = stats["local_routed"] - stats["gate4_fails"]
    total_t1 = stats["tier1_routed"] + stats["gate4_escalate_tier1"]
    total_t2 = stats["tier2_routed"] + stats["gate4_escalate_tier2"]
    
    sys_acc = ((stats["local_correct"] + stats["tier1_correct"] + stats["tier2_correct"]) / total) * 100
    
    saved_toks = stats["baseline_tier2_tokens"] - stats["system_tokens"]
    savings_pct = (saved_toks / max(1, stats["baseline_tier2_tokens"])) * 100
    
    table = Table(title=f"HybridRouter Real Accuracy Benchmark (n={total})")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    
    table.add_row("Final Local Routed (after G4)", f"{total_local} ({total_local/total*100:.1f}%)")
    table.add_row("Final Tier 1 Routed", f"{total_t1} ({total_t1/total*100:.1f}%)")
    table.add_row("Final Tier 2 Routed", f"{total_t2} ({total_t2/total*100:.1f}%)")
    table.add_row("Gate 4 Fails (Rescued to Cloud)", f"{stats['gate4_fails']}")
    table.add_section()
    table.add_row("Local Answer Accuracy", f"{stats['local_correct']}/{max(1, total_local)} ({(stats['local_correct']/max(1, total_local))*100:.1f}%)")
    table.add_row("Tier 1 Answer Accuracy", f"{stats['tier1_correct']}/{max(1, total_t1)} ({(stats['tier1_correct']/max(1, total_t1))*100:.1f}%)")
    table.add_row("Tier 2 Answer Accuracy", f"{stats['tier2_correct']}/{max(1, total_t2)} ({(stats['tier2_correct']/max(1, total_t2))*100:.1f}%)")
    table.add_row("[bold white]Overall System Accuracy[/bold white]", f"[bold white]{sys_acc:.1f}%[/bold white]")
    table.add_section()
    table.add_row("Baseline Tier 2 Tokens", f"{stats['baseline_tier2_tokens']:,}")
    table.add_row("Hybrid System Tokens", f"{stats['system_tokens']:,}")
    table.add_row("[bold yellow]Token Savings[/bold yellow]", f"[bold yellow]{saved_toks:,} ({savings_pct:.1f}%)[/bold yellow]")
    
    CONSOLE.print(table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    run_accuracy_benchmark(args.limit)
