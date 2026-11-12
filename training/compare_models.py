"""
Evaluate every available trained model on the SAME test split and write
reports/model_comparison.csv.

Run from the project root:   python training/compare_models.py
Optional:                    --data-path <csv>

* Models are loaded from models/*.pkl and evaluated with predict() on the notebook's test split.
* A model whose file does not exist is listed as "Not trained" - no numbers are invented.
* Training time is only reported when it was actually measured (KNN / Naive Bayes metrics JSON).
  The Softmax / Random Forest / XGBoost files were trained in the notebook, which did not
  record training time, so those cells show "not recorded".
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import MODEL_DIR, REPORT_DIR, load_split  # noqa: E402
from evaluation import compute_metrics  # noqa: E402

MODELS = [  # (display name, file, report tag, input dtype)
    ("KNN", "knn_model.pkl", "knn", "float32"),
    ("Naive Bayes", "naive_bayes_model.pkl", "naive_bayes", "float32"),
    ("Logistic Regression (Softmax)", "softmax_regression_model.pkl", "softmax", "float32"),
    ("Random Forest", "random_forest_model.pkl", "random_forest", "float32"),
    ("XGBoost", "xgboost_model.pkl", "xgboost", "float32"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default=None)
    args = ap.parse_args()

    _, X_test, _, y_test, _ = load_split(data_path=args.data_path)
    print(f"Test split: {X_test.shape}", flush=True)

    rows = []
    for name, filename, tag, dtype in MODELS:
        path = MODEL_DIR / filename
        row = {"Model": name}
        if not path.exists():
            row.update({"Accuracy": "Not trained", "Notes": f"models/{filename} does not exist"})
            rows.append(row)
            print(f"{name}: not trained", flush=True)
            continue
        try:
            model = joblib.load(path)
        except ModuleNotFoundError as exc:
            row.update({"Accuracy": "Not evaluated", "Notes": f"cannot load model: {exc}"})
            rows.append(row)
            continue

        t0 = time.time()
        y_pred = model.predict(X_test.astype(dtype))
        predict_time = time.time() - t0
        m = compute_metrics(y_test, y_pred)

        train_time = "not recorded"
        notes = ""
        meta_path = REPORT_DIR / f"{tag}_metrics.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if "training_time_seconds" in meta:
                train_time = round(meta["training_time_seconds"], 2)
            if tag == "knn":
                notes = ("KNN has no learning step: training time is only fit() storing the "
                         "data; its cost is paid at prediction time. " +
                         f"Params: {meta.get('best_params')}")
            if tag == "naive_bayes":
                notes = f"Params: {meta.get('best_params')}"
        if tag == "random_forest":
            notes = f"{len(model.estimators_)} trees in the saved file"

        row.update({
            "Accuracy": round(m["accuracy"], 4),
            "Precision (weighted)": round(m["precision_weighted"], 4),
            "Recall (weighted)": round(m["recall_weighted"], 4),
            "F1-score (weighted)": round(m["f1_weighted"], 4),
            "Precision (macro)": round(m["precision_macro"], 4),
            "Recall (macro)": round(m["recall_macro"], 4),
            "F1-score (macro)": round(m["f1_macro"], 4),
            "Training Time (s)": train_time,
            "Prediction Time (s)": round(predict_time, 2),
            "Notes": notes,
        })
        rows.append(row)
        print(f"{name}: accuracy {m['accuracy']:.4f}", flush=True)
        del model

    cols = ["Model", "Accuracy", "Precision (weighted)", "Recall (weighted)",
            "F1-score (weighted)", "Precision (macro)", "Recall (macro)", "F1-score (macro)",
            "Training Time (s)", "Prediction Time (s)", "Notes"]
    out = pd.DataFrame(rows).reindex(columns=cols).fillna("")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    print("\n" + out.drop(columns="Notes").to_string(index=False))


if __name__ == "__main__":
    main()
