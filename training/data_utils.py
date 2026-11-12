"""
Shared data loading for the KNN / comparison scripts.

This is the ONLY place outside the notebook that touches preprocessing. It mirrors
notebooks/02_Model_Training.ipynb exactly:

    1. read cleaned_disease_dataset.csv
    2. drop disease classes that have fewer than 2 rows
    3. LabelEncoder().fit_transform(diseases)
    4. train_test_split(test_size=0.20, random_state=42, stratify=y_encoded)

The LabelEncoder fitted here is compared against models/label_encoder.pkl, so any drift
between the notebook and these scripts is caught immediately.
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dataset" / "cleaned_disease_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

TARGET = "diseases"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def _check_not_lfs_pointer(path):
    with open(path, "rb") as f:
        head = f.read(64)
    if head.startswith(b"version https://git-lfs"):
        raise RuntimeError(
            f"{path.name} is a Git LFS pointer, not the real dataset.\n"
            "Download the real file first, for example:  git lfs pull"
        )


def load_split(feature_dtype="uint8", data_path=None):
    """Return X_train, X_test (DataFrames), y_train, y_test (encoded), label_encoder.

    feature_dtype only changes how the 0/1 columns are stored in RAM (uint8 is 8x smaller
    than the int64 pandas uses by default). The values are identical.
    """
    path = Path(data_path) if data_path else DATA_PATH
    _check_not_lfs_pointer(path)

    columns = pd.read_csv(path, nrows=0).columns
    dtypes = {c: feature_dtype for c in columns if c != TARGET}
    df = pd.read_csv(path, dtype=dtypes)

    class_counts = df[TARGET].value_counts()
    df_model = df[df[TARGET].map(class_counts) >= 2].copy()
    del df

    X = df_model.drop(TARGET, axis=1)
    y = df_model[TARGET]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    saved_path = MODEL_DIR / "label_encoder.pkl"
    if saved_path.exists():
        saved = joblib.load(saved_path)
        if not np.array_equal(saved.classes_, label_encoder.classes_):
            raise RuntimeError(
                "models/label_encoder.pkl does not match the labels rebuilt from the "
                "dataset. Re-run the notebook's label-encoder cell or check the dataset."
            )
        label_encoder = saved   # use the project's own saved encoder

    return X_train, X_test, y_train, y_test, label_encoder


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
