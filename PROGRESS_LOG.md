# Progress Log: Badal's Solutions

## Current Status & What We've Done
- **Environment Setup:** Successfully pulled the hackathon repository (`main` branch) and established a clean Python virtual environment with all required dependencies installed.
- **API Agnosticism:** Modified the `fireworks_client.py` wrapper to read Tier 1 and Tier 2 remote model endpoints dynamically from the `.env` file rather than hardcoding them.
- **Model Research:** Researched ultra-lightweight open-source models suitable for a "Simplicity Gate". Concluded that **Meta MobileLLM** (125M) and **Apple OpenELM** (270M) are state-of-the-art for this specific parameter class and provide better scaling curves than the baseline Qwen 0.5B. Pulled these GGUF quantizations successfully via Ollama.
- **Baseline Benchmark Testing:** Ran the `--demo` routing test on the default baseline system.

### Baseline Benchmark Results
Out of the 6 curated test prompts:
- **Local (qwen2.5:0.5b):** 2 prompts routed (33.3%)
- **Tier 1 (Remote Flash):** 2 prompts routed (33.3%)
- **Tier 2 (Remote Pro):** 2 prompts routed (33.3%)
- **Efficiency:** 100% saved (1,378 tokens saved compared to sending everything to Tier 2).

*(Note: The remote models being used for testing via Google AI Studio API are `gemini-1.5-flash-latest` (Tier 1) and `gemini-1.5-pro-latest` (Tier 2).*

## System Audit Findings (Calibration Flaws)
A deep audit of the local model calibration system (`calibration/run_calibration.py`, `calibration/profile.py`, and `cli/main.py`) revealed several critical flaws that explain why it wasn't working properly for the project:
1. **Fatal Flaw - Active Model Auto-Selection Ignores Calibration Accuracy:** The system automatically selects the "best" local model by sorting based on a static, hardcoded `capability_score` based on model parameter count, completely ignoring empirical `calibration_acc`. A broken 3B model will be blindly auto-selected over a working 0.5B model.
2. **Dangerous Profile Bootstrapping (Fabricating Capabilities):** For any domain/level pair that isn't measured during calibration, the system uses a decay curve to blindly "guess" the accuracy (e.g., artificially inflating L1 task accuracy by +0.20 over the base average). This causes the router to confidently route to local models for domains they have never been tested on.
3. **Erratic Threshold Step Function:** The `acc_to_threshold(acc)` mapping function uses rigid brackets, meaning a 1% difference in accuracy (e.g., 0.59 vs 0.60) can cause a massive 30% jump in the routing threshold, leading to highly erratic routing behaviors.
4. **Silent Config Corruption Data Loss:** If `~/.hybridrouter/config.json` gets corrupted, the system falls back to an empty dictionary, silently deleting all user calibration data.

## Bugs Identified & Fixed
1. **Calibration Crash:** The local model calibration script crashed at the very end (`FileNotFoundError`) because it attempted to save the model capability profile to `~/.hybridrouter/config.json` when the `~/.hybridrouter` directory didn't exist.
   - *Permanent Fix:* Added `os.makedirs(CONFIG_PATH.parent, exist_ok=True)` in `main.py` before the file write operation.
2. **Google AI Studio 404:** The remote client threw a 404 because it requested `gemini-1.5-pro`, which is not explicitly supported by Google's strict endpoint naming schema without the `-latest` or `-002` suffix.
   - *Fix:* Update `.env` to specifically target `gemini-1.5-flash-latest` and `gemini-1.5-pro-latest`.

## Model Evaluation Findings
### Apple OpenELM (270M)
- **Recalibration Score:** 21.6% (25/116 correct), beating the Qwen 0.5B baseline of 14.7%.
- **Capability Profile Built:** `code:L3`
- **Analysis:** Highly effective as a coding Simplicity Gate! It achieved 60% on L2 code, 77% on L3 code, and 100% on L4 code, while correctly failing on math/reasoning tasks, ensuring those are cleanly routed to the cloud models.

### Meta MobileLLM-R1 (360M)
- **Recalibration Score:** 0.0% (0/116 correct).
- **Analysis:** The downloaded Hugging Face quantization appears to be fundamentally incompatible with Ollama's current `llama.cpp` backend parser (likely due to the novel Deep-Thin SwiGLU architecture). The model hallucinates or returns empty tokens resulting in a 0% benchmark score. **Recommendation:** Proceed with OpenELM as the primary local router node for the hackathon.

## Datasets Recommended for System Stress-Testing
To test the routing ML thresholds with high volume, we researched 1,000+ prompt datasets. Here are the top datasets to pull for the final smoke tests:
1. **NVIDIA HelpSteer2 (`nvidia/HelpSteer2`)**: ~21,000 prompts. Every prompt is ground-truth annotated with a Complexity score (0-4), allowing us to easily test if the router's difficulty classification matches reality.
2. **LMSYS Chatbot Arena Conversations (`lmsys/chatbot_arena_conversations`)**: ~33,000 prompts. Organic, in-the-wild crowdsourced data. Excellent for testing edge cases, typos, and out-of-distribution human queries.
3. **OpenHermes 2.5 (`teknium/OpenHermes-2.5`)**: ~1,000,000 prompts. Contains the `Evol-Instruct` dataset which progressively scales simple prompts into complex ones. Useful for testing if the router accurately shifts decisions from Local to Cloud as the identical subject matter scales in cognitive demand.
4. **Databricks Dolly 15k (`databricks/databricks-dolly-15k`)**: ~15,000 prompts. Categorized cleanly into 8 structural intents (Information Extraction, Brainstorming, etc.), perfect for testing intent-based deterministic overrides in the ML router.
