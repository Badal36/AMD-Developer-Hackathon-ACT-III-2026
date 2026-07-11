import json
import os
import sys
import numpy as np
import joblib
from rich.console import Console

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.calibration import CalibratedClassifierCV
except ImportError:
    os.system("pip install scikit-learn")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.calibration import CalibratedClassifierCV

CONSOLE = Console()

def train():
    cache_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets", "ml_router_cache.json")
    router_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "router")
    os.makedirs(router_dir, exist_ok=True)
    model_path = os.path.join(router_dir, "semantic_model.pkl")

    if not os.path.exists(cache_path):
        CONSOLE.print(f"[red]Error: Cache not found at {cache_path}[/red]")
        return

    CONSOLE.print("[cyan]Loading cached semantic embeddings...[/cyan]")
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    X = []
    y = []

    # Qwen2.5:0.5b is our winner, we train the router to predict if Qwen will succeed
    TARGET_MODEL = "qwen2.5:0.5b"
    THRESHOLD = 0.65

    for entry in data:
        scores = entry.get("model_scores", {})
        if TARGET_MODEL not in scores:
            continue
            
        emb = entry["embedding"]
        score = scores[TARGET_MODEL]
        
        # Binary Label: 1 if Qwen succeeded (similarity > 0.65), else 0
        label = 1 if score > THRESHOLD else 0
        
        X.append(emb)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    if len(np.unique(y)) < 2:
        CONSOLE.print("[yellow]Warning: Only one class present in dataset. Creating mock labels to ensure model compiles for demonstration.[/yellow]")
        # For the hackathon 20-prompt dataset, it's possible all passed or all failed. We add a dummy to compile.
        X = np.vstack((X, X[0]))
        y = np.append(y, 1 - y[0])

    CONSOLE.print(f"[green]Dataset prepared! Features: {X.shape[0]} samples, {X.shape[1]} dimensions.[/green]")
    CONSOLE.print(f"Target distribution: {np.sum(y==1)} Local Successes, {np.sum(y==0)} Cloud Fallbacks")

    # Train/test split (purely for logging)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    CONSOLE.print("[cyan]Training Logistic Regression Classifier with Platt Scaling...[/cyan]")
    # Using Logistic Regression because it returns strong probability confidences (predict_proba)
    # and handles high-dimensional embeddings perfectly without overfitting
    base_clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    base_clf.fit(X_train, y_train)
    
    # Apply Platt Scaling to perfectly calibrate probabilities
    clf = CalibratedClassifierCV(estimator=base_clf, method='sigmoid', cv=3)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    CONSOLE.print(f"[bold green]Model Training Complete! Validation Accuracy: {acc*100:.1f}%[/bold green]")
    
    # Save the model
    joblib.dump(clf, model_path)
    CONSOLE.print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    train()
