# GradMit — Graduate Admission Predictor

Full-stack machine learning web application that predicts graduate admission probability using Logistic Regression, achieving high predictive performance (91.2% accuracy, AUC 0.975).

Student: Saanvi Grover
Institute: JIIT Noida
Domain: Data Analytics & Machine Learning

---

## Screenshots
<img width="1920" height="1140" alt="Screenshot 2026-04-05 185322" src="https://github.com/user-attachments/assets/64d2c706-7a6f-4f2b-a25b-a3c695323d02" />

<img width="1920" height="1140" alt="Screenshot 2026-04-05 185338" src="https://github.com/user-attachments/assets/041115fe-8d8f-4d00-a56f-132254a31b37" />

<img width="1920" height="1140" alt="Screenshot 2026-04-05 185359" src="https://github.com/user-attachments/assets/f83a8177-3742-499a-aa9b-4fb691e79caf" />

<img width="1920" height="1140" alt="Screenshot 2026-04-05 185407" src="https://github.com/user-attachments/assets/f7e7bac1-d19e-4f3f-b3b1-68004e72e3c3" />

<img width="1920" height="1140" alt="Screenshot 2026-04-05 185430" src="https://github.com/user-attachments/assets/bbf094fa-5882-41f5-923a-82a10e8d1f18" />

<img width="1920" height="1140" alt="Screenshot 2026-04-05 185416" src="https://github.com/user-attachments/assets/60cb131d-e22c-48cf-911d-a15a0f342c77" />

<img width="1920" height="1140" alt="Screenshot 2026-04-05 185445" src="https://github.com/user-attachments/assets/6d8f0744-9107-4b68-ac8f-ad30755fe24d" />



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

## Author

Saanvi Grover
(AI & Data Analytics)
