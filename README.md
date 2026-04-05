# GradAdmit — Graduate Admission Predictor

**Capstone Project · JIIT Noida · AI & Data Analytics**  
Student: Saanvi Grover | Roll No. 2408030010

A full-stack Flask web application that predicts graduate school admission chances using Logistic Regression (91.2% accuracy, AUC 0.975).

---

## Project Structure

```
gradmit/
├── app.py                    # Flask application — routes, model training, APIs
├── Admission_Predict.csv     # Dataset (500 records from Kaggle)
├── requirements.txt          # Python dependencies
├── README.md
│
├── templates/
│   └── index.html            # Jinja2 HTML template (single-page app)
│
└── static/
    ├── css/
    │   └── style.css         # All styling — dark theme, dashboard, responsive
    └── js/
        └── main.js           # Navigation, API calls, Chart.js, code viewer, table
```

---

## Quick Start

### 1. Install dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Run the app

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET`  | `/`   | Main web app (renders index.html) |
| `POST` | `/api/predict` | Returns admission probability from form values |
| `GET`  | `/api/dataset` | Paginated, filtered dataset records |
| `GET`  | `/api/metrics` | Model performance metrics (accuracy, AUC, etc.) |

### POST `/api/predict` — example

**Request body (JSON):**
```json
{
  "gre": 320,
  "toefl": 110,
  "university_rating": 3,
  "sop": 4.0,
  "lor": 4.0,
  "cgpa": 8.9,
  "research": 1
}
```

**Response:**
```json
{
  "probability": 83.4,
  "admitted": true,
  "contributions": {
    "GRE_Score": 24.13,
    "TOEFL_Score": 14.89,
    ...
  },
  "tips": ["Strong profile! Maintain your academic record."]
}
```

---

## Features

- **Live Predictor** — Sliders send real-time POST requests to Flask; probability updates instantly
- **Analytics Dashboard** — 6 Chart.js charts (ROC, Confusion Matrix, Coefficients, Distribution, CGPA/GRE comparison, Research impact) with filter chips
- **Jupyter Code Viewer** — All notebook cells viewable with syntax highlighting; filter by Code / Markdown
- **Dataset Explorer** — Server-side search, filter (admission status, research), pagination
- **About** — Coefficients table, methodology, future enhancements

---

## Model Details

| Metric | Value |
|--------|-------|
| Algorithm | Logistic Regression |
| Train/Test Split | 75% / 25% (stratified) |
| Accuracy | 91.2% |
| Precision | 92.2% |
| Recall | 87.0% |
| F1 Score | 89.5% |
| AUC | 0.975 |

**Top predictors (by coefficient):**
1. CGPA (+2.4237)
2. Research (+0.5144)
3. SOP (+0.4690)
4. LOR (+0.3529)
5. University Rating (+0.2340)

---

## Production Deployment

```bash
# Using Gunicorn (Linux/macOS)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

For Docker, Nginx, or cloud deployment (Render, Railway, Heroku) — see Flask deployment docs.

---

## Dataset

**Source:** [Kaggle — Graduate Admission Prediction](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions)  
**Records:** 500 student profiles  
**Target:** Chance of Admit ≥ 0.75 → Admitted (binary)
