import json
import time
import os
import sys
import argparse
import numpy as np
from rich.console import Console
from rich.table import Table

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_wrapper.semantic_router import predict_local_viability
from inference_wrapper.local_client import generate, verify_local_response
from inference_wrapper.fireworks_client import call_tier
from sentence_transformers import SentenceTransformer, util

CONSOLE = Console()

def run_smoke_test(dataset_filename="databricks-dolly-15k.jsonl", limit=1000, routing_only=False):
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets", dataset_filename)
    
    if not os.path.exists(dataset_path):
        CONSOLE.print(f"[red]Error: Dataset not found at {dataset_path}[/red]")
        return
        
    prompts_data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit: break
            data = json.loads(line)
            prompt = data.get("instruction", data.get("prompt", ""))
            if data.get("context"):
                prompt += "\nContext: " + data["context"]
            truth = data.get("response", data.get("output", ""))
            prompts_data.append({"prompt": prompt, "truth": truth})

    CONSOLE.print(f"[bold cyan]Loaded {len(prompts_data)} prompts from {dataset_filename}[/bold cyan]")
    if not routing_only:
        CONSOLE.print("[bold cyan]Loading SentenceTransformer (all-MiniLM-L6-v2) for grading...[/bold cyan]")
        try:
            embedder = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            CONSOLE.print(f"[red]Failed to load sentence-transformers: {e}[/red]")
            return
            
        truth_texts = [d["truth"] for d in prompts_data]
        truth_embeddings = embedder.encode(truth_texts, convert_to_tensor=True)

    stats = {
        "router_time": 0.0,
        "local_routed": 0,
        "cloud_routed": 0,
        "local_correct": 0,
        "cloud_correct": 0,
        "local_generation_time": 0.0,
        "cloud_generation_time": 0.0,
        "gate4_fails": 0,
    }

    CONSOLE.print(f"[bold magenta]Starting {'Routing-Only ' if routing_only else ''}Smoke Test on {len(prompts_data)} prompts...[/bold magenta]")
    
    local_model = "qwen2.5:0.5b"

    for i, data in enumerate(prompts_data):
        prompt = data["prompt"]
        
        # 1. Routing Speed & Decision
        t0 = time.time()
        is_local, reason, conf = predict_local_viability(prompt)
        stats["router_time"] += (time.time() - t0)
        
        if routing_only:
            if is_local: stats["local_routed"] += 1
            else: stats["cloud_routed"] += 1
            continue
        
        truth_emb = truth_embeddings[i]
        
        # 2. Generation & Accuracy
        if is_local:
            gen_t0 = time.time()
            raw_response, lat = generate(prompt, local_model, max_tokens=150)
            
            is_valid, _ = verify_local_response(prompt, raw_response)
            if not is_valid:
                stats["gate4_fails"] += 1
                is_local = False # Escalate to cloud
            else:
                stats["local_generation_time"] += (time.time() - gen_t0)
                stats["local_routed"] += 1
                resp_emb = embedder.encode(raw_response, convert_to_tensor=True)
                score = util.cos_sim(truth_emb, resp_emb)[0][0].item()
                if score > 0.65:
                    stats["local_correct"] += 1
        
        if not is_local:
            stats["cloud_routed"] += 1
            gen_t0 = time.time()
            try:
                response, _, _ = call_tier("tier1", prompt)
                stats["cloud_generation_time"] += (time.time() - gen_t0)
                
                resp_emb = embedder.encode(response, convert_to_tensor=True)
                score = util.cos_sim(truth_emb, resp_emb)[0][0].item()
                if score > 0.65:
                    stats["cloud_correct"] += 1
            except Exception as e:
                pass

        if (i + 1) % 50 == 0:
            CONSOLE.print(f"  Processed {i + 1}/{len(prompts_data)}... (Local: {stats['local_routed']}, Cloud: {stats['cloud_routed']})")

    # Output Results
    CONSOLE.print("\n[bold green]Smoke Test Complete![/bold green]")
    
    total = stats["local_routed"] + stats["cloud_routed"]
    avg_router_ms = (stats["router_time"] / max(1, len(prompts_data))) * 1000
    
    local_acc = (stats["local_correct"] / max(1, stats["local_routed"])) * 100
    cloud_acc = (stats["cloud_correct"] / max(1, stats["cloud_routed"])) * 100
    total_acc = ((stats["local_correct"] + stats["cloud_correct"]) / max(1, total)) * 100
    
    table = Table(title=f"Dataset: {dataset_filename} (n={total})")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    
    table.add_row("Avg Routing Speed", f"{avg_router_ms:.2f} ms / prompt")
    table.add_row("Routed to Local", f"{stats['local_routed']} ({stats['local_routed']/max(1,total)*100:.1f}%)")
    table.add_row("Routed to Cloud", f"{stats['cloud_routed']} ({stats['cloud_routed']/max(1,total)*100:.1f}%)")
    
    if not routing_only:
        avg_local_lat = (stats["local_generation_time"] / max(1, stats["local_routed"])) * 1000
        avg_cloud_lat = (stats["cloud_generation_time"] / max(1, stats["cloud_routed"])) * 1000
        table.add_row("Avg Local Latency", f"{avg_local_lat:.0f} ms")
        table.add_row("Avg Cloud Latency", f"{avg_cloud_lat:.0f} ms")
        table.add_row("Local Accuracy (Sim > 0.65)", f"{local_acc:.1f}%")
        table.add_row("Cloud Accuracy (Sim > 0.65)", f"{cloud_acc:.1f}%")
        table.add_row("Total System Accuracy", f"{total_acc:.1f}%")
        table.add_row("Gate 4 Fails (Escalated)", str(stats["gate4_fails"]))
    
    CONSOLE.print(table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="databricks-dolly-15k.jsonl")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--routing-only", action="store_true")
    args = parser.parse_args()
    run_smoke_test(args.dataset, args.limit, args.routing_only)
