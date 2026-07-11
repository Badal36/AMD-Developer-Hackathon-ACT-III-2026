"""
cli/main.py  --  HybridRouter Unified Interactive CLI
------------------------------------------------------
Usage:
  python cli/main.py                          # Interactive mode (recommended)
  python cli/main.py "What is AI?"            # Single-shot prompt
  python cli/main.py --demo                   # Run 6-prompt curated demo
  python cli/main.py --stats                  # Show session stats only
  python cli/main.py --recalibrate            # Recalibrate all models
  python cli/main.py --recalibrate qwen2.5:0.5b  # Recalibrate specific model
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from rich.console  import Console
from rich.panel    import Panel
from rich.table    import Table
from rich.rule     import Rule
from rich.prompt   import Prompt, Confirm
from rich          import box
from rich.markup   import escape

from inference_wrapper.feature_extractor import extract_features
from inference_wrapper.router_core       import predict
from inference_wrapper.simplicity_gate   import is_trivially_simple
from inference_wrapper.local_client      import detect_ollama, score_model, generate as local_gen
from inference_wrapper.fireworks_client  import call_tier, TIER_DISPLAY
from calibration.profile                 import load_profile

CONSOLE     = Console()
CONFIG_PATH = Path.home() / ".hybridrouter" / "config.json"
SESSION_PATH= ROOT / "cli" / ".session.json"

BANNER = r"""
  _   _       _          _     _ ____             _
 | | | |_   _| |__  _ __(_) __| |  _ \ ___  _   _| |_ ___ _ __
 | |_| | | | | '_ \| '__| |/ _` | |_) / _ \| | | | __/ _ \ '__|
 |  _  | |_| | |_) | |  | | (_| |  _ < (_) | |_| | ||  __/ |
 |_| |_|\__, |_.__/|_|  |_|\__,_|_| \_\___/ \__,_|\__\___|_|
         |___/
"""

DEMO_PROMPTS = [
    ("What is 2 + 2?",                                                  "Simple arithmetic"),
    ("If 3x - 7 = 11, find x.",                                         "Algebra"),
    ("What is the capital of France?\nA. Berlin\nB. Paris\nC. Rome\nD. Madrid\nAnswer:", "Science MCQ"),
    ("Write a Python function that reverses a linked list in-place.",    "Code generation"),
    ("Explain the Ship of Theseus paradox and its modern implications.", "Deep reasoning"),
    ("Analyze ethical trade-offs in autonomous AI decision-making.",     "Complex analysis"),
]

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            pass
    return {"models": {}, "active_model": None}

def load_session() -> dict:
    if SESSION_PATH.exists():
        try:
            return json.loads(SESSION_PATH.read_text())
        except Exception:
            pass
    return {"total": 0, "local": 0, "tier1": 0, "tier2": 0,
            "tokens_used": 0, "tokens_saved": 0, "runs": []}

def save_session(s: dict):
    SESSION_PATH.parent.mkdir(exist_ok=True)
    SESSION_PATH.write_text(json.dumps(s, indent=2))

# ---------------------------------------------------------------------------
# Calibration check and prompt
# ---------------------------------------------------------------------------

def check_and_calibrate(cfg: dict, interactive: bool = True) -> dict:
    """
    Detects Ollama models, checks which are calibrated,
    offers to calibrate missing ones. Returns updated config.
    """
    running, models = detect_ollama()
    if not running or not models:
        CONSOLE.print("[yellow]  Ollama not detected — local model gate disabled.[/yellow]")
        CONSOLE.print("  [dim]Start Ollama and run again to enable local routing.[/dim]\n")
        return cfg

    calibrated = cfg.get("models", {})
    all_names  = [m["name"] for m in models]
    uncal      = [n for n in all_names if n not in calibrated]

    # Show status table
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    t.add_column("Local Model",  style="cyan", width=26)
    t.add_column("Cap",          justify="right", width=5)
    t.add_column("Status",       width=30)
    t.add_column("Cal Acc",      justify="right", width=10)
    t.add_column("Threshold",    justify="right", width=12)

    for m in sorted(models, key=lambda x: -score_model(x["name"])):
        name = m["name"]
        cap  = score_model(name)
        if name in calibrated:
            d = calibrated[name]
            t.add_row(name, str(cap),
                      "[green]Calibrated[/green]",
                      f"{d['calibration_acc']*100:.1f}%",
                      str(d["local_threshold"]))
        else:
            t.add_row(name, str(cap),
                      "[yellow]Not calibrated[/yellow]", "--", "--")
    CONSOLE.print(t)

    if not uncal:
        return cfg

    if not interactive:
        # Non-interactive mode (e.g. grading/tests): skip calibration prompt
        return cfg

    CONSOLE.print(f"\n  [yellow]{len(uncal)} model(s) not calibrated.[/yellow]")
    do_cal = Confirm.ask("  Calibrate them now?", default=True)
    if not do_cal:
        CONSOLE.print("  [dim]Skipping calibration. Uncalibrated models won't be used for local routing.[/dim]\n")
        return cfg

    # Run calibration for uncalibrated models
    from calibration.run_calibration import calibrate_model, CAL_PATH
    if not CAL_PATH.exists():
        CONSOLE.print("[red]calibration_prompts.jsonl not found. Run: python calibration/extract_calibration_set.py[/red]")
        return cfg

    prompts = [json.loads(l) for l in CAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    for name in uncal:
        do_this = Confirm.ask(f"  Calibrate [cyan]{name}[/cyan]?", default=True)
        if not do_this:
            continue
        result = calibrate_model(name, prompts)
        cfg.setdefault("models", {})[name] = result
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
        CONSOLE.print(f"  [green]Saved calibration for {name}[/green]")

    # Offer recalibration of already-calibrated models
    already_cal = [n for n in all_names if n in calibrated]
    if already_cal:
        CONSOLE.print()
        redo = Confirm.ask(
            f"  [dim]{len(already_cal)} model(s) already calibrated. Recalibrate any?[/dim]",
            default=False
        )
        if redo:
            for name in already_cal:
                if Confirm.ask(f"  Recalibrate [cyan]{name}[/cyan]?", default=False):
                    result = calibrate_model(name, prompts)
                    cfg["models"][name] = result
                    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
                    CONSOLE.print(f"  [green]Recalibration saved for {name}[/green]")

    return cfg

def recalibrate_model_flow(cfg: dict, target_model: str = None) -> dict:
    """
    Recalibrate one or all models.
    If target_model is given, only recalibrate that model.
    Otherwise, let user pick from a list.
    """
    from calibration.run_calibration import calibrate_model, CAL_PATH

    if not CAL_PATH.exists():
        CONSOLE.print("[red]calibration_prompts.jsonl not found.[/red]")
        return cfg

    running, detected = detect_ollama()
    if not running or not detected:
        CONSOLE.print("[yellow]Ollama not running.[/yellow]")
        return cfg

    all_names      = [m["name"] for m in detected]
    calibrated     = cfg.get("models", {})
    prompts        = [json.loads(l) for l in CAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    if target_model:
        # Validate the model name
        if target_model not in all_names:
            CONSOLE.print(f"[red]Model '{target_model}' not found in Ollama. Available: {all_names}[/red]")
            return cfg
        to_recal = [target_model]
    else:
        # Show menu and let user pick
        CONSOLE.print("\n  [bold]Which model to recalibrate?[/bold]")
        sorted_names = sorted(all_names, key=lambda n: -score_model(n))
        for i, name in enumerate(sorted_names):
            status = f"(acc={calibrated[name]['calibration_acc']*100:.0f}%)" \
                     if name in calibrated else "(not yet calibrated)"
            CONSOLE.print(f"  [{i+1}] [cyan]{name}[/cyan]  {status}")
        CONSOLE.print(f"  [{len(sorted_names)+1}] All models")
        CONSOLE.print(f"  [{len(sorted_names)+2}] Cancel")

        while True:
            choice = Prompt.ask("  Choose", default="1")
            try:
                idx = int(choice) - 1
                if idx == len(sorted_names):
                    to_recal = sorted_names
                    break
                elif idx == len(sorted_names) + 1:
                    return cfg
                elif 0 <= idx < len(sorted_names):
                    to_recal = [sorted_names[idx]]
                    break
            except ValueError:
                pass
            CONSOLE.print("  [red]Invalid choice.[/red]")

    for name in to_recal:
        CONSOLE.print(f"\n  Recalibrating [cyan]{name}[/cyan] with improved scorer...")
        result = calibrate_model(name, prompts)
        cfg.setdefault("models", {})[name] = result
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
        CONSOLE.print(f"  [green]Saved: acc={result['calibration_acc']*100:.1f}%  "
                      f"threshold={result['local_threshold']}[/green]")

    return cfg


def select_active_model(cfg: dict, interactive: bool = True) -> str | None:
    """
    Let user pick which calibrated model to use this session.
    If non-interactive, automatically picks the highest capability calibrated model.
    Returns model name or None (remote-only).
    """
    models = cfg.get("models", {})
    if not models:
        return None

    sorted_models = sorted(models.items(), key=lambda x: -x[1]["capability_score"])

    if not interactive:
        # Auto-select best model for non-interactive execution
        if sorted_models:
            name = sorted_models[0][0]
            CONSOLE.print(f"  [green]Auto-selected local model:[/green] [bold]{name}[/bold]  "
                          f"(cal acc={models[name]['calibration_acc']*100:.1f}%)\n")
            return name
        return None

    if len(sorted_models) == 1:
        name = sorted_models[0][0]
        CONSOLE.print(f"  [green]Using local model:[/green] [bold]{name}[/bold]  "
                      f"(cal acc={models[name]['calibration_acc']*100:.1f}%, "
                      f"threshold={models[name]['local_threshold']})\n")
        return name

    CONSOLE.print("\n  [bold]Select local model for this session:[/bold]")
    for i, (name, data) in enumerate(sorted_models):
        local_pct = int((1 - data["local_threshold"]) * 100)
        CONSOLE.print(f"  [{i+1}] [cyan]{name}[/cyan]  "
                      f"cal={data['calibration_acc']*100:.0f}%  "
                      f"threshold={data['local_threshold']}  "
                      f"~{local_pct}% routed locally")
    CONSOLE.print(f"  [{len(sorted_models)+1}] Remote-only (no local model)")

    while True:
        choice = Prompt.ask("  Choose", default="1")
        try:
            idx = int(choice) - 1
            if idx == len(sorted_models):
                return None
            if 0 <= idx < len(sorted_models):
                return sorted_models[idx][0]
        except ValueError:
            pass
        CONSOLE.print("  [red]Invalid choice.[/red]")

# ---------------------------------------------------------------------------
# Core routing + display
# ---------------------------------------------------------------------------

def route_prompt(prompt: str, active_model: str | None,
                 cfg: dict, session: dict) -> dict:
    """Full routing pipeline with Rich step-by-step display."""

    t_start = time.time()

    # --- Prompt display ---
    CONSOLE.print()
    CONSOLE.print(Rule("[bold white]Prompt[/bold white]", style="dim white"))
    CONSOLE.print(Panel(
        escape(prompt[:500]) + ("..." if len(prompt) > 500 else ""),
        border_style="white", padding=(0, 1)
    ))

    # =========================================================================
    # STEP 0: Feature Extraction
    # =========================================================================
    t_feat_start = time.time()
    feats = extract_features(prompt)
    fms = (time.time() - t_feat_start) * 1000

    # Load capability profile for active model (None if not calibrated)
    capability_profile = load_profile(active_model) if active_model else None

    # =========================================================================
    # STEP 1: Gate 0 -- Simplicity Pre-Filter
    # =========================================================================
    CONSOLE.print(Rule("[bold green]Step 1  Gate 0 -- Capability-Aware Routing[/bold green]", style="green"))

    # Compute model context first, then call gate
    has_local  = bool(active_model and cfg.get("models", {}).get(active_model))

    model_data = cfg.get("models", {}).get(active_model, {}) if active_model else {}
    model_acc  = model_data.get("calibration_acc", 0.0)
    src_stats  = model_data.get("source_stats", {})

    from inference_wrapper.simplicity_gate import _local_threshold
    gate_threshold = _local_threshold(model_acc) if has_local else "n/a"

    t0 = time.time()
    is_simple, gate_reason, gate_conf = is_trivially_simple(
        prompt, feats, model_acc,
        src_stats if has_local else None,
        capability_profile=capability_profile if has_local else None,
    )
    gms = (time.time() - t0) * 1000

    # Detect which routing path was used
    using_profile = (capability_profile is not None and has_local
                     and gate_reason.startswith("Profile ["))
    routing_mode  = "[cyan]Profile (calibrated)[/cyan]" if using_profile \
                    else "[dim]Legacy heuristic[/dim]"

    gt = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    gt.add_column("Label", style="dim", width=22)
    gt.add_column("Value", width=60)
    gt.add_row("Routing mode",    routing_mode)
    gt.add_row("Decision",        gate_reason)
    gt.add_row("Confidence",      f"[bold]{gate_conf:.2f}[/bold]")
    gt.add_row("Local model",     f"[cyan]{active_model or 'None'}[/cyan]")
    if using_profile and capability_profile:
        gt.add_row("Profile summary", f"[dim]{capability_profile.summary()}[/dim]")
    else:
        gt.add_row("Model cal acc",   f"{model_acc*100:.1f}%" if has_local else "n/a")
    gt.add_row("Gate latency",    f"{gms:.2f}ms (feat extract: {fms:.1f}ms)")
    CONSOLE.print(gt)

    dest         = "local"
    tokens_used  = 0
    tokens_saved = 0
    response     = ""

    if is_simple and has_local:
        # ── Simple prompt + local model available: route directly to local ──
        CONSOLE.print(Panel(
            f"[bold green]Gate 0: SIMPLE -> LOCAL  ({active_model})[/bold green]\n"
            f"[dim]Bypassing ML router. 0 Fireworks tokens.[/dim]",
            border_style="green", padding=(0, 2)
        ))
        CONSOLE.print(Rule("[bold white]Inference  (Local)[/bold white]", style="green"))
        CONSOLE.print(f"  [green]Calling Ollama ({active_model})...[/green]", end="")
        response, latency = local_gen(prompt, active_model)
        tokens_baseline   = max(len(response.split()) * 3, 150)
        tokens_saved      = tokens_baseline
        CONSOLE.print(f"  done in {latency:.2f}s")
        dest = "local"

    else:
        # ── Not simple (or no local model): ML router decides tier1 vs tier2 ──
        if is_simple and not has_local:
            CONSOLE.print(Panel(
                "[yellow]Gate 0: SIMPLE but no local model calibrated -> REMOTE[/yellow]\n"
                "[dim]Run --recalibrate to enable local routing.[/dim]",
                border_style="yellow", padding=(0, 2)
            ))
        else:
            CONSOLE.print(Panel(
                "[cyan]Gate 0: NOT SIMPLE -> ML Router[/cyan]\n"
                "[dim]Prompt requires remote model intelligence.[/dim]",
                border_style="cyan", padding=(0, 2)
            ))

        # =================================================================
        # STEP 2: ML Router -- tier1 vs tier2
        # =================================================================
        CONSOLE.print(Rule("[bold magenta]Step 2  ML Router -- Tier Decision[/bold magenta]", style="magenta"))
        t0   = time.time()
        tier, t1p, t2p = predict(feats)
        rms  = (time.time() - t0) * 1000

        bar_width = 30
        p1b = "#" * int(t1p * bar_width) + "-" * (bar_width - int(t1p * bar_width))
        p2b = "#" * int(t2p * bar_width) + "-" * (bar_width - int(t2p * bar_width))
        CONSOLE.print(f"  [green]Tier1 gpt-oss-20b  [{p1b}] {t1p:.3f}[/green]")
        CONSOLE.print(f"  [yellow]  Tier2 glm-5p2     [{p2b}] {t2p:.3f}[/yellow]")
        CONSOLE.print(f"  [dim]Features: task={feats['_task_type']}  "
                      f"complexity={feats['complexity_heuristic']}  "
                      f"depth={feats['llm_reasoning_depth']}  "
                      f"Router latency: {rms:.1f}ms[/dim]")

        if tier == "tier1":
            CONSOLE.print(Panel(
                "[bold cyan]Router: TIER 1  (gpt-oss-20b)[/bold cyan]\n"
                "[dim]Moderate complexity -- cheap fast model.[/dim]",
                border_style="cyan", padding=(0, 2)
            ))
            dest = "tier1"
        else:
            CONSOLE.print(Panel(
                "[bold yellow]Router: TIER 2  (glm-5p2)[/bold yellow]\n"
                "[dim]High complexity -- powerful model required.[/dim]",
                border_style="yellow", padding=(0, 2)
            ))
            dest = "tier2"

        # =================================================================
        # STEP 3: Remote Inference
        # =================================================================
        CONSOLE.print(Rule("[bold white]Step 3  Inference  (Remote)[/bold white]", style="white"))
        if dest == "tier1":
            CONSOLE.print(f"  [cyan]Calling Fireworks Tier 1 (gpt-oss-20b)...[/cyan]", end="")
            response, tokens_used, latency = call_tier("tier1", prompt)
            tokens_baseline = max(int(tokens_used * 2.8), tokens_used + 200)
            CONSOLE.print(f"  done in {latency:.2f}s  [{tokens_used} tokens]")
        else:
            CONSOLE.print(f"  [yellow]Calling Fireworks Tier 2 (glm-5p2)...[/yellow]", end="")
            response, tokens_used, latency = call_tier("tier2", prompt)
            tokens_baseline = tokens_used
            CONSOLE.print(f"  done in {latency:.2f}s  [{tokens_used} tokens]")

        tokens_saved = max(tokens_baseline - tokens_used, 0)

    total_ms = (time.time() - t_start) * 1000

    # --- Response ---
    CONSOLE.print(Rule("[bold white]Response[/bold white]", style="white"))
    color = "green" if dest == "local" else "cyan" if dest == "tier1" else "yellow"
    CONSOLE.print(Panel(
        escape(response[:1800]) + ("..." if len(response) > 1800 else ""),
        border_style=color, padding=(0, 1)
    ))

    # --- Token summary ---
    st = Table(box=box.ROUNDED, border_style=color, show_header=False)
    st.add_column("", style="bold", width=22)
    st.add_column("", width=44)
    dest_label = (f"Local  ({active_model})" if dest == "local"
                  else "Tier 1  gpt-oss-20b" if dest == "tier1"
                  else "Tier 2  glm-5p2")
    gate_label = "Gate 0 (simple)" if dest == "local" else "ML Router"
    st.add_row("Decision by",      gate_label)
    st.add_row("Routed to",        f"[{color}]{dest_label}[/{color}]")
    st.add_row("Fireworks tokens", f"[bold]{tokens_used}[/bold]")
    st.add_row("Tokens saved",     f"[bold green]+{tokens_saved}[/bold green]  vs remote baseline")
    st.add_row("Total latency",    f"{total_ms:.0f}ms")
    CONSOLE.print(st)

    # Update session
    session["total"]        += 1
    session[dest]           += 1
    session["tokens_used"]  += tokens_used
    session["tokens_saved"] += tokens_saved
    session["runs"].append({"dest": dest, "tokens": tokens_used, "saved": tokens_saved})
    return {
        "dest": dest,
        "tokens": tokens_used,
        "saved": tokens_saved,
        "response": response,
        "latency_ms": round(total_ms, 1)
    }

# ---------------------------------------------------------------------------
# Stats display
# ---------------------------------------------------------------------------

def show_stats(session: dict):
    CONSOLE.print()
    CONSOLE.print(Rule("[bold cyan]Session Statistics[/bold cyan]", style="cyan"))
    total = max(session["total"], 1)
    st = Table(title="Routing Summary", box=box.ROUNDED, border_style="cyan")
    st.add_column("Destination",  style="bold")
    st.add_column("Count",        justify="right")
    st.add_column("Share",        justify="right")
    st.add_column("Avg Tokens",   justify="right")
    for dest, label, color in [
        ("local", "Local (Ollama)", "green"),
        ("tier1", "Tier 1  gpt-oss-20b", "cyan"),
        ("tier2", "Tier 2  glm-5p2", "yellow"),
    ]:
        n    = session.get(dest, 0)
        runs = [r for r in session["runs"] if r["dest"] == dest]
        avg  = int(sum(r["tokens"] for r in runs) / max(len(runs), 1))
        if n:
            st.add_row(
                f"[{color}]{label}[/{color}]",
                str(n), f"{n/total*100:.1f}%",
                "[green]0[/green]" if dest == "local" else str(avg)
            )
    CONSOLE.print(st)
    tok   = session["tokens_used"]
    saved = session["tokens_saved"]
    pct   = saved / max(tok + saved, 1) * 100
    CONSOLE.print(Panel(
        f"[bold]Fireworks Tokens Used:[/bold]  [cyan]{tok}[/cyan]\n"
        f"[bold]Tokens Saved:[/bold]           [green]+{saved}[/green]  vs always-tier2 baseline\n"
        f"[bold]Efficiency:[/bold]             [bold green]{pct:.1f}% saved[/bold green]\n"
        f"[bold]Prompts Handled:[/bold]        {session['total']}",
        title="[bold green]Cost Summary[/bold green]",
        border_style="green"
    ))

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = [a for a in sys.argv[1:] if a]

    is_json = "--json" in args
    if is_json:
        args.remove("--json")
        CONSOLE.quiet = True

    # Determine if we are in interactive mode
    # It is interactive only if no demo, stats, recalibrate, or single-shot prompt is requested
    has_demo = "--demo" in args
    has_stats = "--stats" in args
    has_recal = "--recalibrate" in args
    has_singleshot = len(args) > 0 and not args[0].startswith("--")

    is_interactive = not (has_demo or has_stats or has_recal or has_singleshot)

    # Print banner (only prints if CONSOLE.quiet is False)
    CONSOLE.print(f"[bold cyan]{BANNER}[/bold cyan]")
    CONSOLE.print(Panel.fit(
        "[bold]AMD Developer Hackathon ACT II 2026[/bold]  --  Token-Efficient LLM Routing\n"
        "[dim]Routes each prompt to the cheapest model that can answer it correctly.[/dim]",
        border_style="cyan"
    ))

    # Stats-only mode
    if "--stats" in args:
        show_stats(load_session())
        return

    CONSOLE.print()
    CONSOLE.print(Rule("[bold cyan]Startup[/bold cyan]", style="cyan"))

    # Load config + detect + calibrate
    cfg = load_config()

    # Handle --recalibrate [model] flag BEFORE normal startup
    if "--recalibrate" in args:
        idx         = args.index("--recalibrate")
        target      = args[idx + 1] if idx + 1 < len(args) and not args[idx+1].startswith("--") else None
        cfg         = recalibrate_model_flow(cfg, target)
        active_model = select_active_model(cfg, interactive=is_interactive)
        session      = load_session()
        CONSOLE.print("[green]Recalibration complete. Starting interactive mode...[/green]\n")
        # Fall through to interactive mode with updated config
    else:
        cfg          = check_and_calibrate(cfg, interactive=is_interactive)
        active_model = select_active_model(cfg, interactive=is_interactive)
        session      = load_session()

    # Demo mode
    if "--demo" in args:
        CONSOLE.print(Panel(
            "[bold]Curated Demo[/bold] -- 6 prompts across all difficulty levels\n"
            "[dim]Watch routing decisions: local / tier1 / tier2[/dim]",
            border_style="cyan"
        ))
        for prompt, label in DEMO_PROMPTS:
            CONSOLE.print(f"\n  [dim italic]{label}[/dim italic]")
            route_prompt(prompt, active_model, cfg, session)
            time.sleep(0.3)
        show_stats(session)
        save_session(session)
        return

    # Single-shot CLI argument mode
    if has_singleshot:
        prompt = " ".join(args)
        res = route_prompt(prompt, active_model, cfg, session)
        if is_json:
            # Output pure, clean JSON on standard stdout, bypassing Rich output entirely
            sys.stdout.write(json.dumps(res, indent=2) + "\n")
            sys.stdout.flush()
        else:
            show_stats(session)
        save_session(session)
        return

    # Interactive loop
    CONSOLE.print()
    CONSOLE.print(Panel(
        "[bold]Interactive Mode[/bold]\n"
        "Type your prompt and press Enter to route it.\n\n"
        "Commands:\n"
        "  [bold cyan]stats[/bold cyan]                  -- session token summary\n"
        "  [bold cyan]switch[/bold cyan]                 -- change active local model\n"
        "  [bold cyan]recalibrate[/bold cyan]            -- recalibrate a model (pick from list)\n"
        "  [bold cyan]recalibrate <model>[/bold cyan]    -- recalibrate a specific model\n"
        "  [bold cyan]exit[/bold cyan] / [bold cyan]quit[/bold cyan]           -- exit",
        border_style="cyan", padding=(0, 2)
    ))

    while True:
        CONSOLE.print()
        try:
            prompt = Prompt.ask("[bold cyan]>>>[/bold cyan] Prompt")
        except (KeyboardInterrupt, EOFError):
            break

        if not prompt.strip():
            continue
        cmd = prompt.strip().lower()
        if cmd in ("exit", "quit", "q"):
            break
        if cmd == "stats":
            show_stats(session)
            continue
        if cmd == "switch":
            active_model = select_active_model(cfg)
            continue
        if cmd.startswith("recalibrate"):
            parts       = prompt.strip().split(maxsplit=1)
            target      = parts[1] if len(parts) > 1 else None
            cfg         = recalibrate_model_flow(cfg, target)
            # Refresh active model in case threshold changed
            model_data  = cfg.get("models", {}).get(active_model, {})
            if active_model and model_data:
                new_thr = model_data.get("local_threshold", 1.0)
                new_acc = model_data.get("calibration_acc", 0.0)
                CONSOLE.print(f"  [green]Active model updated: acc={new_acc*100:.1f}%  threshold={new_thr}[/green]")
            continue

        route_prompt(prompt.strip(), active_model, cfg, session)

    show_stats(session)
    save_session(session)
    CONSOLE.print("\n[dim]Session saved. Goodbye.[/dim]")


if __name__ == "__main__":
    main()
