import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model https://drive.google.com/file/d/1jEEJFekKusTcAyb7RPvd3K_j8EA4q9UN/view?usp=sharing 
import gdown
import os
os.environ["GDOWN_CACHE_DIR"] = os.path.expanduser("~/.gdown_cache")
# Google Drive model link 
url = "https://drive.google.com/uc?id=1jEEJFekKusTcAyb7RPvd3K_j8EA4q9UN"
output = "final_loan_model.joblib"

# Download only once
if not os.path.exists(output):
    gdown.download(url, output, quiet=False, use_cookies=False)

model = joblib.load(output)


st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰")

# Title
st.title("🏦 Loan Approval Risk Predictor")
st.write("This app predicts whether a loan applicant is **Safe** or **High Risk** based on financial data.")

# Sidebar Inputs
st.sidebar.header("Applicant Financial Details")

income = st.sidebar.number_input("Applicant Income (₹)", min_value=0, value=50000, step=1000)
loan_amount = st.sidebar.number_input("Loan Amount (₹)", min_value=0, value=200000, step=1000)
property_value = st.sidebar.number_input("Property Value (₹)", min_value=0, value=300000, step=1000)
rate_of_interest = st.sidebar.number_input("Rate of Interest (%)", min_value=1.0, max_value=20.0, value=8.5, step=0.1)
upfront_charges = st.sidebar.number_input("Upfront Charges (₹)", min_value=0, value=1500, step=100)
term = st.sidebar.number_input("Loan Term (months)", min_value=12, max_value=360, value=240, step=12)

# Feature Engineering (same as in your notebook)
loan_to_income_ratio = loan_amount / (income + 1)
LTV_percent = (loan_amount / (property_value + 1)) * 100
interest_burden = (rate_of_interest * loan_amount) / (income + 1)
emi_to_income_ratio = interest_burden / 10
upfront_charge_percent = (upfront_charges / (loan_amount + 1)) * 100
long_term_flag = 1 if term >= 240 else 0

# DataFrame for model input
features = pd.DataFrame([{
    'loan_to_income_ratio': loan_to_income_ratio,
    'LTV_percent': LTV_percent,
    'interest_burden': interest_burden,
    'emi_to_income_ratio': emi_to_income_ratio,
    'upfront_charge_percent': upfront_charge_percent,
    'long_term_flag': long_term_flag
}])

# Prediction
if st.button("🔍 Predict Loan Risk"):
    # Get predicted probabilities
    probabilities = model.predict_proba(features)[0]
    risk_prob = probabilities[1]  # Probability of being risky

    # Custom threshold (tune this if needed)
    threshold = 0.6  

    if risk_prob > threshold:
        st.error(f"⚠️ High Risk Borrower — Probability: {risk_prob:.2f}")
    else:
        st.success(f"✅ Safe Borrower — Probability: {risk_prob:.2f}")

    # Display feature values and probability
    st.subheader("Feature Values Used for Prediction:")
    st.write(features)