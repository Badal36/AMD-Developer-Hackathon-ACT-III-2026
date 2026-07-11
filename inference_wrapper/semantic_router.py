import os
import joblib
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass

_model = None
_embedder = None

def get_router_model():
    global _model
    if _model is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "router", "semantic_model.pkl")
        if os.path.exists(model_path):
            _model = joblib.load(model_path)
    return _model

def get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            pass
    return _embedder

def predict_local_viability(prompt: str) -> tuple[bool, str, float]:
    """
    Predicts if the local model (Qwen2.5:0.5b) can successfully answer the prompt.
    Returns:
        (is_viable: bool, reason: str, confidence: float)

    Confidence threshold raised to 0.70 (from 0.60).
    The local model only gets the task if the router is STRONGLY confident
    it can handle it. This prevents the local model from winning on ambiguous
    prompts where it produces garbage — those now route to cloud instead.

    Effect on routing:
      conf >= 0.70  ->  Route LOCAL  (strong prediction, trusted)
      conf <  0.70  ->  Route CLOUD  (uncertain, fail-safe to cloud)
    """
    model = get_router_model()
    embedder = get_embedder()

    if model is None or embedder is None:
        return False, "ML Router unavailable (missing .pkl or sentence-transformers)", 0.0

    emb = embedder.encode(prompt)
    emb = np.array(emb).reshape(1, -1)

    # 0 = Fallback to Cloud, 1 = Route Local
    prediction = model.predict(emb)[0]
    probabilities = model.predict_proba(emb)[0]

    confidence = float(probabilities[1])  # Probability of class 1 (Local Success)

    LOCAL_CONFIDENCE_FLOOR = 0.70
    if confidence >= LOCAL_CONFIDENCE_FLOOR:
        return True, f"ML Prediction [Local] (conf: {confidence:.2f} >= {LOCAL_CONFIDENCE_FLOOR})", confidence
    else:
        return False, f"ML Prediction [Cloud] (conf: {confidence:.2f} < {LOCAL_CONFIDENCE_FLOOR} floor)", confidence
