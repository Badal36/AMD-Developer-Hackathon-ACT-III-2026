import json
import os
from pathlib import Path

def main():
    config_path = Path.home() / ".hybridrouter" / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {"models": {}, "active_model": "hf.co/mradermacher/OpenELM-270M-GGUF:latest"}
        
    cfg["active_model"] = "hf.co/mradermacher/OpenELM-270M-GGUF:latest"
    
    # Mock OpenELM profile
    cfg["models"]["hf.co/mradermacher/OpenELM-270M-GGUF:latest"] = {
        "calibration_acc": 0.216,
        "capability_profile": {
            "code": {
                "L1": 0.95,
                "L2": 0.85
            },
            "factual": {
                "L1": 0.98,
                "L2": 0.91,
                "L3": 0.40
            },
            "math": {
                "L1": 0.50
            },
            "language": {
                "L1": 0.80
            },
            "reasoning": {
                "L1": 0.70
            },
            "instruction": {
                "L1": 0.75
            }
        }
    }
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        
    print("Mock profile injected!")

if __name__ == "__main__":
    main()
