"""Shared evaluation + report writing (used by train_knn, compare_models)."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

from data_utils import REPORT_DIR, save_json


def compute_metrics(y_true, y_pred):
    """Accuracy plus weighted and macro precision / recall / F1 (zero_division=0)."""
    out = {"accuracy": accuracy_score(y_true, y_pred)}
    for avg in ("weighted", "macro"):
        out[f"precision_{avg}"] = precision_score(y_true, y_pred, average=avg, zero_division=0)
        out[f"recall_{avg}"] = recall_score(y_true, y_pred, average=avg, zero_division=0)
        out[f"f1_{avg}"] = f1_score(y_true, y_pred, average=avg, zero_division=0)
    return {k: float(v) for k, v in out.items()}


def write_reports(tag, y_true, y_pred, label_encoder, extra=None):
    """Write metrics JSON, full classification report, confusion matrix CSV and top confusions."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    labels = np.arange(len(label_encoder.classes_))
    names = list(label_encoder.classes_)

    metrics = compute_metrics(y_true, y_pred)
    if extra:
        metrics.update(extra)
    save_json(metrics, REPORT_DIR / f"{tag}_metrics.json")

    report = classification_report(y_true, y_pred, labels=labels, target_names=names,
                                   zero_division=0)
    (REPORT_DIR / f"{tag}_classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=names, columns=names).to_csv(
        REPORT_DIR / f"{tag}_confusion_matrix.csv")

    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(off, axis=None)[::-1][:30]
    rows = []
    for idx in flat:
        i, j = divmod(int(idx), off.shape[1])
        if off[i, j] > 0:
            rows.append({"true_disease": names[i], "predicted_disease": names[j],
                         "count": int(off[i, j])})
    pd.DataFrame(rows).to_csv(REPORT_DIR / f"{tag}_top_confusions.csv", index=False)
    return metrics
