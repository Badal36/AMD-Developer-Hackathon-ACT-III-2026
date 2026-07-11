import json
import time
import os
import sys
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_wrapper.local_client import generate, verify_local_response
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer, util
import numpy as np

CONSOLE = Console()

def run_benchmark():
    num_prompts = 20
    models = [
        "hf.co/mradermacher/OpenELM-270M-GGUF:latest",
        "smollm2:135m",
        "smollm2:360m",
        "qwen2.5:0.5b"
    ]
    
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets", "databricks-dolly-15k.jsonl")
    
    if not os.path.exists(dataset_path):
        CONSOLE.print(f"[red]Error: Dataset not found at {dataset_path}[/red]")
        return
        
    prompts_data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_prompts: break
            data = json.loads(line)
            # Combine instruction and context for the prompt
            prompt = data["instruction"]
            if data.get("context"):
                prompt += "\nContext: " + data["context"]
                
            prompts_data.append({
                "prompt": prompt,
                "truth": data["response"],
                "category": data["category"]
            })
            
    CONSOLE.print(f"[bold cyan]Loading SentenceTransformer (all-MiniLM-L6-v2) for grading...[/bold cyan]")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    CONSOLE.print(f"[bold cyan]Pre-calculating ground truth embeddings...[/bold cyan]")
    truth_texts = [d["truth"] for d in prompts_data]
    truth_embeddings = embedder.encode(truth_texts, convert_to_tensor=True)
    
    results = defaultdict(lambda: {"correct": 0, "gate4_fails": 0, "total_time": 0.0, "total_tokens_saved": 0})
    
    # We invert the loops here to prevent Ollama from constantly loading/unloading models from memory
    # By looping model-first, Ollama keeps the model loaded for all 20 prompts, drastically speeding it up.
    
    ml_router_cache = [{"prompt": d["prompt"], "category": d["category"], "embedding": embedder.encode(d["prompt"]).tolist(), "model_scores": {}} for d in prompts_data]
    
    CONSOLE.print(f"[bold magenta]Starting Benchmark: 4 models × {num_prompts} prompts = {4 * num_prompts} inferences...[/bold magenta]")
    
    for model in models:
        CONSOLE.print(f"[yellow]Evaluating model: {model}[/yellow]")
        for i, data in enumerate(prompts_data):
            prompt = data["prompt"]
            truth_emb = truth_embeddings[i]
            
            # We limit to 150 tokens to speed up the benchmark
            raw_response, latency = generate(prompt, model, max_tokens=150)
            
            results[model]["total_time"] += latency
            
            # Step 1: Does it pass Gate 4?
            is_valid, reason = verify_local_response(prompt, raw_response)
            
            score = 0.0
            if not is_valid:
                results[model]["gate4_fails"] += 1
            else:
                # Step 2: Semantic Similarity
                resp_emb = embedder.encode(raw_response, convert_to_tensor=True)
                cosine_scores = util.cos_sim(truth_emb, resp_emb)
                score = cosine_scores[0][0].item()
                
                if score > 0.65:
                    results[model]["correct"] += 1
                    results[model]["total_tokens_saved"] += 200
            
            ml_router_cache[i]["model_scores"][model] = score
            
            if (i + 1) % 10 == 0:
                CONSOLE.print(f"  Processed {i + 1}/{num_prompts} for {model}...")
            
    # Dump cache for Phase 3
    cache_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets", "ml_router_cache.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(ml_router_cache, f)
        
    CONSOLE.print("\n[bold green]Benchmark Complete![/bold green]")
    
    table = Table(title=f"Local Model Output Quality (n={num_prompts})")
    table.add_column("Model", justify="left", style="cyan")
    table.add_column("Accuracy (Sim > 0.65)", justify="right", style="green")
    table.add_column("Gate 4 Fails", justify="right", style="red")
    table.add_column("Avg Latency (s)", justify="right", style="yellow")
    table.add_column("Tokens Saved", justify="right", style="magenta")
    
    for model in models:
        stats = results[model]
        acc = (stats["correct"] / num_prompts) * 100
        avg_lat = stats["total_time"] / num_prompts
        table.add_row(
            model,
            f"{acc:.1f}%",
            str(stats["gate4_fails"]),
            f"{avg_lat:.2f}s",
            f"{stats['total_tokens_saved']:,}"
        )
        
    CONSOLE.print(table)

if __name__ == "__main__":
    run_benchmark()
