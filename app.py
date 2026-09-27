import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Load Model + Preprocessing Objects
best_model = load_model("best_model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Numeric columns used during training
numeric_cols = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_percent_income",
    "cb_person_cred_hist_length"
]

grade_order = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7}

# Streamlit UI

st.title("📌 Loan Default Risk Prediction App")
st.write("Enter customer details below to predict the loan default risk.")

# Input Form
with st.form("customer_form"):

    st.subheader("🔍 Customer Information")

    person_age = st.number_input("Age", 18, 100, 30)
    person_income = st.number_input("Annual Income (USD)", 10000, 300000, 60000)
    person_emp_length = st.number_input("Employment Length (years)", 0.0, 50.0, 5.0)

    st.subheader("🏡 Home & Loan Details")

    # MORTGAGE and DEBTCONSOLIDATION were the dropped (baseline) categories
    # in one-hot encoding, so they map to all-zero dummy columns
    person_home_ownership = st.selectbox(
        "Home Ownership",
        ["RENT", "OWN", "MORTGAGE", "OTHER"]
    )

    loan_intent = st.selectbox(
        "Loan Intent",
        ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE", "DEBTCONSOLIDATION"]
    )

    loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])

    loan_amnt = st.number_input("Loan Amount (USD)", 500, 50000, 12000)
    loan_percent_income = st.number_input("Loan Percent of Income", 0.01, 1.0, 0.20)

    st.subheader("📈 Credit History")

    cb_default = st.selectbox("Customer Defaulted Before?", ["N", "Y"])
    cb_person_cred_hist_length = st.number_input(
        "Credit History Length (years)", 0, 40, 7
    )

    submit = st.form_submit_button("Predict Risk")


# Prediction Logic
if submit:

    # ---- Build DataFrame from input ----
    new_customer = pd.DataFrame([{
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_default,
        "cb_person_cred_hist_length": cb_person_cred_hist_length
    }])

    # ---- Match training preprocessing ----
    new_customer["cb_person_default_on_file"] = new_customer["cb_person_default_on_file"].map({"Y": 1, "N": 0})
    new_customer["loan_grade"] = new_customer["loan_grade"].map(grade_order)

    # One-hot encoding. No drop_first here: with a single row it would drop
    # the only category. The reindex below removes the baseline columns.
    new_customer = pd.get_dummies(
        new_customer,
        columns=["person_home_ownership", "loan_intent"]
    )

    # Align columns with training data
    new_customer = new_customer.reindex(columns=feature_columns, fill_value=0)

    # Scale numeric columns
    existing_numeric = [col for col in numeric_cols if col in new_customer.columns]
    new_customer[existing_numeric] = scaler.transform(new_customer[existing_numeric])

    # ---- Predict ----
    probability = best_model.predict(new_customer)[0][0]
    prediction = 1 if probability > 0.5 else 0

    risk_label = "🔴 HIGH-RISK" if prediction == 1 else "🟢 LOW-RISK"

    # ---- Display Output ----
    st.subheader("📌 Prediction Result")
    st.markdown(f"### **Risk Classification: {risk_label}**")
    st.write(f"**Probability of Default:** `{probability:.4f}`")
