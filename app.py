
import os
import joblib
import pandas as pd

from collections import Counter
from flask import Flask, render_template, request


app = Flask(__name__)

# --------------------------------------------------
# 1. MODEL FILE PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

softmax_model = joblib.load(
    os.path.join(MODEL_DIR, "softmax_regression_model.pkl")
)

random_forest_model = joblib.load(
    os.path.join(MODEL_DIR, "random_forest_model.pkl")
)

knn_model = joblib.load(
    os.path.join(MODEL_DIR, "knn_model.pkl")
)

naive_bayes_model = joblib.load(
    os.path.join(MODEL_DIR, "naive_bayes_model.pkl")
)

label_encoder = joblib.load(
    os.path.join(MODEL_DIR, "label_encoder.pkl")
)


# --------------------------------------------------
# 2. STORE ALL FOUR MODELS
# --------------------------------------------------

MODELS = {
    "Softmax Regression": softmax_model,
    "Random Forest": random_forest_model,
    "KNN": knn_model,
    "Bernoulli Naive Bayes": naive_bayes_model
}


# --------------------------------------------------
# 3. GET SYMPTOM FEATURES
# --------------------------------------------------

if not hasattr(softmax_model, "feature_names_in_"):
    raise ValueError(
        "Softmax model does not contain feature names. "
        "Train it using a pandas DataFrame."
    )

FEATURE_NAMES = list(softmax_model.feature_names_in_)

for model_name, model in MODELS.items():
    if hasattr(model, "feature_names_in_"):
        if list(model.feature_names_in_) != FEATURE_NAMES:
            raise ValueError(
                f"{model_name} has different feature names or order."
            )

SYMPTOM_OPTIONS = [
    {
        "value": symptom,
        "label": symptom.replace("_", " ").replace("-", " ").title()
    }
    for symptom in FEATURE_NAMES
]


# --------------------------------------------------
# 4. PREPARE USER INPUT
# --------------------------------------------------

def prepare_input(selected_symptoms):

    valid_symptoms = set(FEATURE_NAMES)

    # Remove invalid and duplicate symptoms
    selected_symptoms = list(dict.fromkeys(
        symptom for symptom in selected_symptoms
        if symptom in valid_symptoms
    ))

    if len(selected_symptoms) < 2:
        raise ValueError("Please select at least two symptoms.")

    # Create a row with all symptom values initially set to 0
    input_data = pd.DataFrame(
        [[0] * len(FEATURE_NAMES)],
        columns=FEATURE_NAMES
    )

    # Set selected symptoms to 1
    for symptom in selected_symptoms:
        input_data.loc[0, symptom] = 1

    return input_data, selected_symptoms


# --------------------------------------------------
# 5. CONVERT PREDICTION TO DISEASE NAME
# --------------------------------------------------

def decode_prediction(prediction):

    prediction = int(prediction)

    return str(
        label_encoder.inverse_transform([prediction])[0]
    )


# --------------------------------------------------
# 6. MAJORITY VOTING
# --------------------------------------------------

def majority_voting(input_data):

    predictions = {}
    vote_counts = Counter()
    probability_totals = Counter()

    # Every model makes its prediction
    for model_name, model in MODELS.items():

        prediction = model.predict(input_data)[0]

        disease = decode_prediction(prediction)

        predictions[model_name] = disease

        vote_counts[disease] += 1

    # Find the maximum number of votes
    maximum_votes = max(vote_counts.values())

    # Find diseases having the maximum votes
    winners = [
        disease
        for disease, votes in vote_counts.items()
        if votes == maximum_votes
    ]

    # One clear majority winner
    if len(winners) == 1:
        final_disease = winners[0]

    else:
        # Resolve a tie using model probabilities
        for model in MODELS.values():

            if not hasattr(model, "predict_proba"):
                continue

            probabilities = model.predict_proba(input_data)[0]

            model_classes = model.classes_

            for class_id, probability in zip(
                model_classes, probabilities
            ):

                disease = decode_prediction(class_id)

                if disease in winners:
                    probability_totals[disease] += float(probability)

        if probability_totals:
            final_disease = sorted(
                winners,
                key=lambda disease: (
                    -probability_totals[disease],
                    disease
                )
            )[0]

        else:
            # Deterministic fallback
            final_disease = sorted(winners)[0]

    return final_disease, predictions, dict(vote_counts)


# --------------------------------------------------
# 7. HOME PAGE
# --------------------------------------------------

def render_home(
    selected_symptoms=None,
    error=None,
    final_disease=None,
    predictions=None,
    vote_counts=None
):

    return render_template(
        "index.html",
        symptom_options=SYMPTOM_OPTIONS,
        selected_symptoms=selected_symptoms or [],
        error=error,
        final_disease=final_disease,
        predictions=predictions,
        vote_counts=vote_counts
    )


# --------------------------------------------------
# 8. FLASK ROUTES
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        symptom_text = request.form.get(
            "symptoms", ""
        ).strip()

        selected_symptoms = [
            symptom.strip()
            for symptom in symptom_text.split(",")
            if symptom.strip()
        ]

        try:

            # Prepare symptom input
            input_data, selected_symptoms = prepare_input(
                selected_symptoms
            )

            # Get final prediction using majority voting
            final_disease, predictions, vote_counts = (
                majority_voting(input_data)
            )

            return render_home(
                selected_symptoms=selected_symptoms,
                final_disease=final_disease,
                predictions=predictions,
                vote_counts=vote_counts
            )

        except ValueError as error:

            return render_home(
                selected_symptoms=selected_symptoms,
                error=str(error)
            )

        except Exception as error:

            app.logger.exception("Prediction failed")

            return render_home(
                selected_symptoms=selected_symptoms,
                error="Prediction failed. Check the model files and input features."
            )

    return render_home()


# --------------------------------------------------
# 9. RUN FLASK APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
