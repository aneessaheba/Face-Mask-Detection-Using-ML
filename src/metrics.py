import json
from typing import Dict, List, Tuple

import numpy as np


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict[str, object]:
    y_true = np.array(y_true).astype(int)
    y_pred_prob = np.array(y_pred)
    y_pred_bin = (y_pred_prob >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred_bin == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred_bin == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred_bin == 0)))

    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "confusion_matrix": [[int(tp), int(fp)], [int(fn), int(tn)]],
    }


def save_metrics_report(metrics: Dict[str, object], output_path: str) -> str:
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return output_path
