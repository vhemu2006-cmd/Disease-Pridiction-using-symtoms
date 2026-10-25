from flask import Flask, render_template, request
from pathlib import Path
import joblib
import pandas as pd
import re

app = Flask(__name__)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Load trained models
softmax_model = joblib.load(
    MODEL_DIR / "softmax_regression_model.pkl"
)

random_forest_model = joblib.load(
    MODEL_DIR / "random_forest_model.pkl"
)

label_encoder = joblib.load(
    MODEL_DIR / "label_encoder.pkl"
)


def normalize_symptom(value):
    """Normalize symptom names for matching."""
    value = str(value).strip().lower()
    value = re.sub(r"[\s-]+", "_", value)
    return value


def load_feature_names():
    """Get the symptom columns in the order used for training."""

    if hasattr(softmax_model, "feature_names_in_"):
        softmax_features = list(softmax_model.feature_names_in_)

        if hasattr(random_forest_model, "feature_names_in_"):
            forest_features = list(
                random_forest_model.feature_names_in_
            )

            if softmax_features != forest_features:
                raise ValueError(
                    "The two models have different feature columns "
                    "or feature orders."
                )

        return softmax_features

    if hasattr(random_forest_model, "feature_names_in_"):
        return list(random_forest_model.feature_names_in_)

    raise ValueError(
        "Model feature names are unavailable. "
        "The exact training feature columns must be loaded."
    )


FEATURE_NAMES = load_feature_names()

# Match normalized names to the exact model feature names.
FEATURE_LOOKUP = {
    normalize_symptom(feature): feature
    for feature in FEATURE_NAMES
}

# Dropdown options are generated from the actual model features.
SYMPTOM_OPTIONS = sorted(
    [
        {
            "value": feature,
            "label": feature.replace("_", " ").strip().title()
        }
        for feature in FEATURE_NAMES
    ],
    key=lambda symptom: symptom["label"].casefold()
)


def prepare_input(symptom_text):
    """Convert the textbox into the model's expected input."""

    entered_symptoms = [
        item.strip()
        for item in symptom_text.split(",")
        if item.strip()
    ]

    if not entered_symptoms:
        raise ValueError("Please select or enter at least one symptom.")

    matched_features = set()
    unknown_symptoms = []

    for symptom in entered_symptoms:
        normalized = normalize_symptom(symptom)

        if normalized in FEATURE_LOOKUP:
            matched_features.add(FEATURE_LOOKUP[normalized])
        else:
            unknown_symptoms.append(symptom)

    if not matched_features:
        raise ValueError(
            "None of the entered symptoms matched the model features. "
            "Please select symptoms from the dropdown or use their "
            "exact feature names."
        )

    # Keep the exact feature order used during training.
    input_row = {
        feature: int(feature in matched_features)
        for feature in FEATURE_NAMES
    }

    input_data = pd.DataFrame(
        [input_row],
        columns=FEATURE_NAMES
    )

    return input_data, unknown_symptoms


def get_disease_prediction(model, input_data):
    """Predict a disease and decode its label."""

    prediction = model.predict(input_data)[0]

    # The label encoder must correspond to the models' target labels.
    disease = label_encoder.inverse_transform([prediction])[0]

    return str(disease)


def render_home(**kwargs):
    """Render the page with the available symptom options."""
    return render_template(
        "index.html",
        symptom_options=SYMPTOM_OPTIONS,
        **kwargs
    )


@app.route("/", methods=["GET"])
def home():
    return render_home()


@app.route("/predict", methods=["POST"])
def predict():
    symptoms = request.form.get("symptoms", "").strip()

    if not symptoms:
        return render_home(
            error="Please select or enter your symptoms."
        )

    try:
        input_data, unknown_symptoms = prepare_input(symptoms)

        softmax_prediction = get_disease_prediction(
            softmax_model,
            input_data
        )

        random_forest_prediction = get_disease_prediction(
            random_forest_model,
            input_data
        )

        warning = None

        if unknown_symptoms:
            warning = (
                "These symptoms were not recognized and were ignored: "
                + ", ".join(unknown_symptoms)
            )

        return render_home(
            symptoms=symptoms,
            softmax_prediction=softmax_prediction,
            random_forest_prediction=random_forest_prediction,
            warning=warning
        )

    except ValueError as exc:
        return render_home(
            symptoms=symptoms,
            error=str(exc)
        )

    except Exception:
        app.logger.exception("Prediction error")

        return render_home(
            symptoms=symptoms,
            error=(
                "Prediction failed. Check the model files, "
                "feature columns, and label encoder."
            )
        ), 500


if __name__ == "__main__":
    app.run(debug=True)