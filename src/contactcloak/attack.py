import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def read(paths):
    rows = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            rows.extend(json.loads(line)["result"] for line in stream if line.strip())
    x = np.asarray([row["features"] for row in rows], dtype=float)
    y = np.asarray([row["attack_label"] for row in rows])
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Attack features must be a finite fixed-width matrix")
    return x, y


def auc(model, x, y):
    probabilities = model.predict_proba(x)
    if len(model.classes_) == 2:
        return float(roc_auc_score(y == model.classes_[1], probabilities[:, 1]))
    return float(roc_auc_score(y, probabilities, labels=model.classes_, multi_class="ovr"))


def main():
    parser = argparse.ArgumentParser()
    for name in ("train", "validation", "test"):
        parser.add_argument("--" + name, nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--fpr", type=float, default=0.01)
    args = parser.parse_args()
    x, y = read(args.train)
    vx, vy = read(args.validation)
    tx, ty = read(args.test)
    if not set(np.unique(ty)) <= set(np.unique(y)) or len(np.unique(y)) < 2:
        raise ValueError("Training must contain every evaluated secret class")
    candidates = [
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=args.seed)),
        ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=2, random_state=args.seed, n_jobs=-1
        ),
    ]
    trained = [model.fit(x, y) for model in candidates]
    validation_scores = [auc(model, vx, vy) for model in trained]
    best = trained[int(np.argmax(validation_scores))]
    prediction = best.predict(tx)
    report = {
        "top1": float(accuracy_score(ty, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(ty, prediction)),
        "auc": auc(best, tx, ty),
        "validation_auc": max(validation_scores),
        "train_samples": len(y),
        "validation_samples": len(vy),
        "test_samples": len(ty),
    }
    if len(best.classes_) == 2:
        positive = best.classes_[1]
        negative_scores = best.predict_proba(vx)[vy != positive, 1]
        threshold = float(np.quantile(negative_scores, 1 - args.fpr, method="higher"))
        scores = best.predict_proba(tx)[:, 1]
        report["threshold"] = threshold
        report["tpr"] = (
            float(np.mean(scores[ty == positive] > threshold)) if np.any(ty == positive) else None
        )
        report["fpr"] = (
            float(np.mean(scores[ty != positive] > threshold)) if np.any(ty != positive) else None
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, output / "attacker.joblib")
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
