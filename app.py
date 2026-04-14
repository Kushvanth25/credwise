import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# Page Config
st.set_page_config(
    page_title="CredWise",
    page_icon="💳",
    layout="wide"
)

# Title
st.title("💳 CredWise - AI Loan Risk Prediction")
st.caption("AI-powered Loan Risk & Approval Prediction System")

st.divider()

# Sidebar Inputs
st.sidebar.header("Applicant Details")

income = st.sidebar.number_input("Applicant Income", value=10000.0)
loan_amount = st.sidebar.number_input("Loan Amount", value=100000.0)
credit_score = st.sidebar.number_input("Credit Score", value=650.0)
loan_term = st.sidebar.number_input("Loan Term (Months)", value=12)
dti = st.sidebar.number_input("DTI Ratio", value=0.5)

employer = st.sidebar.selectbox(
    "Employer Category",
    ["MNC", "Other"]
)

marital = st.sidebar.selectbox(
    "Marital Status",
    ["Single", "Married"]
)

st.sidebar.divider()

predict = st.sidebar.button("Predict Risk")

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Applicant Summary")
    st.write(f"**Income:** ₹{income}")
    st.write(f"**Loan Amount:** ₹{loan_amount}")
    st.write(f"**Credit Score:** {credit_score}")
    st.write(f"**Loan Term:** {loan_term} months")
    st.write(f"**DTI Ratio:** {dti}")

# Prediction
if predict:

    input_data = pd.DataFrame([[0]*len(columns)], columns=columns)

    input_data["Applicant_Income"] = income
    input_data["Loan_Amount"] = loan_amount
    input_data["Loan_Term"] = loan_term

    # Feature Engineering
    input_data["DTI_Ratio_sq"] = dti**2
    input_data["Credit_Score_sq"] = credit_score**2

    if employer == "MNC":
        input_data["Employer_Category_MNC"] = 1

    if marital == "Single":
        input_data["Marital_Status_Single"] = 1

    # Scale
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]
    risk = 1 - probability
    risk_percent = risk * 100

    with col2:
        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

        # Risk Score
        st.subheader("Risk Score")

        st.progress(float(risk))
        st.write(f"Risk Score: {risk_percent:.0f}%")

        # Credit Grade
        if risk < 0.2:
            grade = "A"
        elif risk < 0.4:
            grade = "B"
        elif risk < 0.7:
            grade = "C"
        else:
            grade = "D"

        st.metric("Credit Grade", grade)

    # Risk Factors
    st.divider()
    st.subheader("Risk Factors")

    risk_factors = []

    if dti > 0.6:
        risk_factors.append("High Debt-to-Income Ratio")

    if credit_score < 600:
        risk_factors.append("Low Credit Score")

    if loan_amount > income * 5:
        risk_factors.append("Loan amount too high compared to income")

    if len(risk_factors) == 0:
        st.success("No major risk factors detected")
    else:
        for factor in risk_factors:
            st.warning(f"⚠️ {factor}")