# Credit Risk Decision Engine

## Overview

The Credit Risk Decision Engine is a Machine Learning project that predicts whether a loan applicant is likely to have **Good Credit** or **Bad Credit** based on their financial and personal information.

The project includes an end-to-end ML pipeline:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Model training and evaluation
* Model comparison
* FastAPI-based prediction API

---

## Dataset

* **Dataset:** German Credit Dataset
* **Source:** UCI Machine Learning Repository
* **Instances:** 1000
* **Target Variable:** Credit Risk (Good Credit / Bad Credit)

---

## Project Workflow

1. Load and explore the dataset.
2. Perform Exploratory Data Analysis (EDA).
3. Handle categorical variables using One-Hot Encoding.
4. Split data into training and testing sets.
5. Scale numerical features using StandardScaler.
6. Train multiple machine learning models.
7. Compare model performance.
8. Save the best-performing model using Joblib.
9. Build a FastAPI application for real-time predictions.

---

## Models Evaluated

| Model               | Accuracy  |
| ------------------- | --------- |
| Logistic Regression | **79.5%** |
| Random Forest       | 75.0%     |
| XGBoost             | 78.5%     |

### Final Model

**Logistic Regression** was selected because it achieved the best overall performance on the German Credit dataset.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Joblib
* FastAPI
* Uvicorn

---

## Project Structure

```text
Credit_risk/
│
├── app.py
├── models/
│   ├── credit_risk_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── notebooks/
│   └── credit_risk.ipynb
│
├── data/
│   └── german.data
│
├── README.md
└── requirements.txt
```

---

## Running the Project

Clone the repository:

```bash
git clone <repository-url>
cd Credit_risk
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn app:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

to test the API using Swagger UI.

---

## Example API Request

```json
{
  "checking_account": "A12",
  "duration": 12,
  "credit_history": "A32",
  "purpose": "A43",
  "credit_amount": 5000,
  "savings": "A61",
  "employment": "A73",
  "installment_rate": 2,
  "personal_status_sex": "A93",
  "other_debtors": "A101",
  "residence_since": 3,
  "property": "A121",
  "age": 35,
  "other_installment_plans": "A143",
  "housing": "A152",
  "existing_credits": 1,
  "job": "A173",
  "dependents": 1,
  "telephone": "A191",
  "foreign_worker": "A201"
}
```

Example Response:

```json
{
  "prediction": "Bad Credit"
}
```

---

## Future Improvements

* Return prediction confidence scores.
* Improve feature engineering.
* Add input validation for categorical values.
* Deploy the API to a cloud platform.
* Build a web interface for predictions.

---

## Author

**Jaya Sai Pranathi**

Computer Science Engineering Student | AI & Machine Learning Enthusiast
