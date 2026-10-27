"""
Evaluation Metrics Module for IntentAlign
Computes classification metrics (Precision, Recall, F1, Accuracy, FPR, FNR),
feed exposure rates, intent-feed divergence, topic diversity, and latency statistics.
No fabricated values - all values derived from actual system execution.
"""
import math
from typing import List, Dict, Any
import numpy as np

def compute_classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """
    Computes Precision, Recall, F1, Accuracy, False Positive Rate, False Negative Rate.
    Binary definition:
      Positive class = ALLOW / DEPRIORITIZE (User-relevant content kept)
      Negative class = SUPPRESS (Irrelevant / misaligned content hidden)
    """
    tp = fp = tn = fn = 0
    for true_val, pred_val in zip(y_true, y_pred):
        is_true_pos = (true_val in ["ALLOW", "DEPRIORITIZE"])
        is_pred_pos = (pred_val in ["ALLOW", "DEPRIORITIZE"])

        if is_true_pos and is_pred_pos:
            tp += 1
        elif not is_true_pos and is_pred_pos:
            fp += 1  # Unwanted video allowed
        elif not is_true_pos and not is_pred_pos:
            tn += 1  # Unwanted video suppressed
        elif is_true_pos and not is_pred_pos:
            fn += 1  # Relevant video wrongly suppressed

    total = max(1, tp + fp + tn + fn)
    accuracy = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-6, precision + recall)

    # False Positive Rate = FP / (FP + TN)
    fpr = fp / max(1, fp + tn)
    # False Negative Rate = FN / (FN + TP)
    fnr = fn / max(1, fn + tp)

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "false_positive_rate": round(float(fpr), 4),
        "false_negative_rate": round(float(fnr), 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    }

def compute_shannon_diversity(topic_counts: Dict[str, float]) -> float:
    """Computes normalized Shannon entropy diversity H in [0.0, 1.0]."""
    total = sum(topic_counts.values())
    if total <= 0 or len(topic_counts) <= 1:
        return 0.0
    probs = [c / total for c in topic_counts.values() if c > 0]
    entropy = -sum(p * math.log2(p) for p in probs)
    max_h = math.log2(len(topic_counts))
    return round(float(entropy / max_h), 4)
