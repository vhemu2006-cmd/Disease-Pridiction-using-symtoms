# KNN and Naive Bayes additions

KNN and Naive Bayes are in `notebooks/02_Model_Training.ipynb` (cells 29-32, after XGBoost). Run those cells
to train and save `models/knn_model.pkl` and `models/naive_bayes_model.pkl`. (AdaBoost was tried and removed.)

## Files
```
models/knn_model.pkl, knn_best_params.json
models/naive_bayes_model.pkl, naive_bayes_best_params.json
reports/                 metrics JSON, classification reports, confusion matrices (CSV), tuning table (KNN),
                         model_comparison.csv
training/                optional helper scripts (KNN tuning, comparison table). app.py does not need them.
app.py, templates/index.html, static/style.css   additive edits: "Additional model" dropdown
```
Both models reuse `models/label_encoder.pkl`. No scaler is needed (features are 0/1). Feature names and order
are stored in each model (`feature_names_in_`) and `app.py` checks that all models use the same order.

## Install and run
1. `pip install -r requirements.txt` (and `pip install xgboost` if you want the XGBoost model/row).
2. `dataset/*.csv` must be the real files, not Git LFS pointers: `git lfs pull`.
3. Run the notebook cells, then `python app.py` and open http://127.0.0.1:5000.
4. Optional comparison table: `python training/compare_models.py` (writes `reports/model_comparison.csv`).

## Results (same test split for every model; from reports/model_comparison.csv)
| Model | Accuracy | F1 (weighted) | F1 (macro) |
|---|---|---|---|
| Naive Bayes (BernoulliNB, alpha=0.001, uniform class prior) | 0.8402 | 0.8474 | 0.7430 |
| Logistic Regression (Softmax) | 0.8318 | 0.8331 | 0.7271 |
| KNN (k=7, uniform, cosine) | 0.8108 | 0.8092 | 0.7041 |
| Random Forest (8 trees in the saved file) | 0.7919 | 0.7902 | 0.5669 |
| XGBoost | not trained | | |

Naive Bayes was chosen against KNN, an MLP neural network and other Naive Bayes variants using a validation split
and 5-fold StratifiedKFold of the training data only; the test split was used once for the final score.
Its probabilities (and KNN's vote share) are shown in the app as "Model probability"; Naive Bayes
probabilities tend to be overconfident, so treat them as a rough guide.
Models were trained with scikit-learn 1.8.0 (KNN) and 1.9.1 (Naive Bayes); use the same scikit-learn version when loading.
