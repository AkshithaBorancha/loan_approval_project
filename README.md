# 🏦 Loan Approval Risk Predictor

This project predicts whether a loan applicant is **Safe** or **High Risk** based on their financial data using **Machine Learning** and a hybrid **rule-based + AI** approach.

---

## 🚀 Live App

🔗 [**Try the App on Streamlit**](https://loan-approval-project-akshithaborancha.streamlit.app)

---

## 📊 Project Overview

Banks face challenges in identifying safe vs. risky loan applicants.  
This project builds a **Loan Approval Risk Prediction System** that analyzes applicant financial information to assess loan risk.

---

## 🧠 Features of the Project

✅ Trained on a **Loan Default Dataset (1.4 lakh+ records)**  
✅ Engineered 6+ financial ratio features for better insight  
✅ Balanced data using **SMOTE**  
✅ Deployed as an **interactive Streamlit web app**  
✅ Uses a **hybrid of AI prediction + financial logic** for realistic decisions  

---

## 📁 Dataset Used

- **Dataset Name:** Loan_Default.csv  
- **Source:** Open Microsoft / Kaggle Loan Risk datasets  
- **Rows:** 1,48,000+  
- **Columns:** 34  
- **Target Variable:** `high_risk` → (1 = Risky borrower, 0 = Safe borrower)

---

## 🔍 Feature Engineering

| Feature | Formula | Description |
|----------|----------|-------------|
| `loan_to_income_ratio` | `loan_amount / income` | Measures loan relative to income |
| `LTV_percent` | `(loan_amount / property_value) * 100` | Loan-to-Value ratio |
| `interest_burden` | `(interest_rate * loan_amount) / income` | Borrower’s interest load |
| `emi_to_income_ratio` | `interest_burden / 10` | Approximate EMI pressure |
| `upfront_charge_percent` | `(upfront_charges / loan_amount) * 100` | Upfront cost ratio |
| `long_term_flag` | `1 if term >= 240 else 0` | Flags long-term loans |

---

## 🤖 Model Training

- **Algorithm:** Random Forest Classifier  
- **Data Balancing:** SMOTE (Synthetic Minority Oversampling Technique)  
- **Accuracy:** 95%  
- **Weighted F1-score:** 0.94  
- **Frameworks:** scikit-learn, pandas, numpy

---

## ☁️ Model Storage (Google Drive)

Since GitHub restricts file sizes >100 MB,  
the trained model (`final_loan_model.joblib`, 330 MB) is hosted on **Google Drive**.

It is **automatically downloaded** when the app runs using `gdown`:

```python
import gdown, joblib, os

url = "https://drive.google.com/uc?id=1jEEJFekKusTcAyb7RPvd3K_j8EA4q9UN"
output = "final_loan_model.joblib"

if not os.path.exists(output):
    gdown.download(url, output, quiet=False, use_cookies=False)

model = joblib.load(output)
