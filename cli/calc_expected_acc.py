import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli.main import load_config
from calibration.profile import load_profile
from inference_wrapper.difficulty_classifier import classify
from inference_wrapper.feature_extractor import extract_features

def main():
    config = load_config()
    active_model = config.get("active_model", "hf.co/mradermacher/OpenELM-270M-GGUF:latest")
    cap_profile = load_profile(active_model)
    
    dataset_path = "../datasets/databricks-dolly-15k.jsonl"
    prompts = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 1000: break
            data = json.loads(line)
            prompt = data["instruction"]
            if data.get("context"): prompt += "\nContext: " + data["context"]
            prompts.append(prompt)
            
    expected_correct = 0.0
    routed_local = 0
    
    for prompt in prompts:
        feats = extract_features(prompt)
        difficulty = classify(prompt, feats)
        domain = difficulty.domain
        level_str = f"L{difficulty.level}"
        
        route_local, reason = cap_profile.should_route_local(domain, level_str)
        if route_local:
            routed_local += 1
            acc = cap_profile.acc.get(domain, {}).get(level_str)
            if acc is None:
                # If a specific level is missing but it routed local, it means max_level covered it. 
                # Estimate it using the confidence floor (worst case)
                acc = 0.65
            expected_correct += float(acc)
            
    if routed_local > 0:
        overall_acc = (expected_correct / routed_local) * 100
        print(f"Routed Local: {routed_local}")
        print(f"Expected Correct: {expected_correct:.1f}")
        print(f"Expected Wrong: {routed_local - expected_correct:.1f}")
        print(f"Estimated Local Accuracy: {overall_acc:.1f}%")

if __name__ == "__main__":
    main()
