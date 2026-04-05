"""
GradAdmit — Graduate Admission Prediction
Flask Application
Author: Saanvi Grover | JIIT Noida | AI & Data Analytics Capstone
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import os

app = Flask(__name__)

# ─── Load & Train Model on Startup ──────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Admission_Predict.csv")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.replace(" ", "_")
if "Serial_No." in df.columns:
    df.drop(columns=["Serial_No."], inplace=True)

df["Admitted"] = (df["Chance_of_Admit"] >= 0.75).astype(int)
df.drop(columns=["Chance_of_Admit"], inplace=True)

FEATURES = ["GRE_Score", "TOEFL_Score", "University_Rating", "SOP", "LOR", "CGPA", "Research"]
X = df[FEATURES]
y = df["Admitted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Pre-compute metrics
cm = confusion_matrix(y_test, y_pred).tolist()
fpr_arr, tpr_arr, _ = roc_curve(y_test, y_proba)

METRICS = {
    "accuracy":  round(accuracy_score(y_test, y_pred) * 100, 1),
    "precision": round(precision_score(y_test, y_pred) * 100, 1),
    "recall":    round(recall_score(y_test, y_pred) * 100, 1),
    "f1":        round(f1_score(y_test, y_pred) * 100, 1),
    "auc":       round(roc_auc_score(y_test, y_proba), 3),
    "confusion_matrix": cm,
    "roc": {
        "fpr": [round(v, 4) for v in fpr_arr.tolist()],
        "tpr": [round(v, 4) for v in tpr_arr.tolist()],
    },
    "coefficients": {
        feat: round(coef, 4)
        for feat, coef in zip(FEATURES, model.coef_[0])
    },
    "class_counts": {
        "admitted":     int(y.sum()),
        "not_admitted": int((y == 0).sum()),
    },
    "group_stats": {
        "cgpa_admitted":     round(df[df.Admitted == 1]["CGPA"].mean(), 2),
        "cgpa_rejected":     round(df[df.Admitted == 0]["CGPA"].mean(), 2),
        "gre_admitted":      round(df[df.Admitted == 1]["GRE_Score"].mean(), 2),
        "gre_rejected":      round(df[df.Admitted == 0]["GRE_Score"].mean(), 2),
        "res_admitted_rate": round(df[df.Admitted == 1]["Research"].mean() * 100, 1),
        "res_rejected_rate": round(df[df.Admitted == 0]["Research"].mean() * 100, 1),
    },
}

# Dataset records for table view
RECORDS = df.reset_index(drop=True).to_dict(orient="records")


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", metrics=METRICS)


@app.route("/api/predict", methods=["POST"])
def predict():
    """Accept form values, return probability and feature contributions."""
    data = request.get_json(force=True)
    try:
        vals = [
            float(data["gre"]),
            float(data["toefl"]),
            float(data["university_rating"]),
            float(data["sop"]),
            float(data["lor"]),
            float(data["cgpa"]),
            int(data["research"]),
        ]
        import pandas as _pd
        _df_vals = _pd.DataFrame([vals], columns=FEATURES)
        prob = float(model.predict_proba(_df_vals)[0][1])
        admitted = prob >= 0.5

        # Contribution = coeff * value (normalised to 0-1 for the bar)
        raw_contribs = {
            feat: round(coef * val, 4)
            for feat, coef, val in zip(FEATURES, model.coef_[0], vals)
        }

        tips = []
        if vals[5] < 8.5:   tips.append("Aim for CGPA above 8.5 — the strongest predictor.")
        if vals[6] == 0:    tips.append("Research experience significantly boosts chances.")
        if vals[3] < 4:     tips.append("Strengthen your SOP — aim for 4.0+.")
        if vals[0] < 320:   tips.append("GRE above 320 aligns with admitted students.")
        if not tips:        tips.append("Strong profile! Maintain your record and research involvement.")

        return jsonify({
            "probability": round(prob * 100, 1),
            "admitted": admitted,
            "contributions": raw_contribs,
            "tips": tips,
        })
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/dataset")
def dataset():
    """Return paginated, filtered dataset records."""
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 15))
    search   = request.args.get("search", "").lower()
    admitted = request.args.get("admitted", "")
    research = request.args.get("research", "")

    rows = RECORDS
    if search:
        rows = [r for r in rows if any(search in str(v).lower() for v in r.values())]
    if admitted != "":
        rows = [r for r in rows if str(r["Admitted"]) == admitted]
    if research != "":
        rows = [r for r in rows if str(r["Research"]) == research]

    total   = len(rows)
    start   = (page - 1) * per_page
    records = rows[start : start + per_page]

    return jsonify({"total": total, "page": page, "records": records})


@app.route("/api/metrics")
def metrics():
    return jsonify(METRICS)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
