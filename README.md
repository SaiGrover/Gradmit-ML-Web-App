# GradMit — Graduate Admission Predictor

Full-stack machine learning web application that predicts graduate admission probability using Logistic Regression, achieving high predictive performance (91.2% accuracy, AUC 0.975).

Student: Saanvi Grover
Institute: JIIT Noida
Domain: Data Analytics & Machine Learning

---

## Overview

GradMit helps students estimate their chances of admission based on academic and profile features like GRE, TOEFL, CGPA, SOP, LOR, and Research experience.

The system combines data preprocessing, model training, and an interactive Flask-based web interface to deliver real-time predictions and insights.

---

## Key Features

* Real-time admission prediction using trained ML model
* Interactive dashboard with 6+ visualizations (ROC, Confusion Matrix, Feature Importance, etc.)
* Dataset explorer with filtering, search, and pagination
* Contribution analysis showing feature impact on predictions
* Clean and responsive UI using Flask, HTML, CSS, and JavaScript

---

## Tech Stack

* Python
* Flask
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Chart.js (frontend visualization)

---

## Model Performance

* Algorithm: Logistic Regression
* Accuracy: 91.2%
* Precision: 92.2%
* Recall: 87.0%
* F1 Score: 89.5%
* AUC Score: 0.975

Key Insight: CGPA and Research experience are the strongest predictors of admission probability.

---

## Dataset

* Source: Kaggle – Graduate Admission Dataset
* Size: 500 records
* Target: Admission (binary classification based on probability threshold)

---

## Project Structure

```
gradmit/
│── app.py
│── data/
│   └── Admission_Predict.csv
│── templates/
│── static/
│── requirements.txt
│── README.md
```
---

## How to Run

1. Clone the repository

2. Install dependencies
   pip install -r requirements.txt

3. Run the application
   python app.py

4. Open in browser
   http://localhost:5000

---

## API Endpoints

POST /api/predict → Returns admission probability
GET /api/dataset → Returns dataset with filters
GET /api/metrics → Returns model performance

---

## Results & Impact

* Built an end-to-end ML pipeline from data preprocessing to deployment
* Achieved 91%+ prediction accuracy on unseen data
* Enabled real-time decision support for students applying to graduate programs
* Improved interpretability using feature contribution analysis

---

## Future Improvements

* Deploy application using cloud platforms (Render / Railway)
* Add multiple ML models for comparison (Random Forest, XGBoost)
* Improve UI/UX with advanced dashboards
* Integrate user login and history tracking

---

## Screenshots

(Updating soon)

---

## Author

Saanvi Grover
(AI & Data Analytics)
