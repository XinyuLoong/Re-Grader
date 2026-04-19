import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"

PRED_PATH = RESULTS_DIR / "predictions.json"
METRICS_PATH = RESULTS_DIR / "metrics.json"


def safe_div(n, d):
    return n / d if d else 0.0


def main():
    with open(PRED_PATH, "r", encoding="utf-8") as f:
        preds = json.load(f)

    tp = fp = fn = tn = 0
    abstain_count = 0

    for row in preds:
        gt = row["ground_truth_label"]          # consistent / inconsistent
        pred = row["prediction"]                # consistent / inconsistent / abstain

        if pred == "abstain":
            abstain_count += 1

        if gt == "inconsistent" and pred == "inconsistent":
            tp += 1
        elif gt == "consistent" and pred == "inconsistent":
            fp += 1
        elif gt == "inconsistent" and pred != "inconsistent":
            fn += 1
        elif gt == "consistent" and pred == "consistent":
            tn += 1

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    accuracy = safe_div(tp + tn, len(preds))
    false_positive_rate = safe_div(fp, fp + tn)
    abstain_rate = safe_div(abstain_count, len(preds))

    metrics = {
        "num_cases": len(preds),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision_inconsistent": round(precision, 3),
        "recall_inconsistent": round(recall, 3),
        "accuracy": round(accuracy, 3),
        "false_positive_rate": round(false_positive_rate, 3),
        "abstain_rate": round(abstain_rate, 3),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()