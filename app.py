import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the model and scaler
model = joblib.load('insurance_cost_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title
st.title("Medical Insurance Cost Predictor 🏥")

# Input widgets
st.header("Patient Information")
age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", ["Female", "Male"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["No", "Yes"])
region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

# Preprocessing
sex_encoded = 1 if sex == "Male" else 0
smoker_encoded = 1 if smoker == "Yes" else 0

# One-hot encode region (MUST match Phase 2 preprocessing!)
region_encoded = {
    "Northeast": [0, 0, 0],
    "Northwest": [1, 0, 0],
    "Southeast": [0, 1, 0],
    "Southwest": [0, 0, 1]
}[region]

# Scale numerical features (age, bmi, children)
scaled_features = scaler.transform([[age, bmi, children]])

# Create feature array
features = np.array([
    scaled_features[0][0],  # scaled age
    sex_encoded,
    scaled_features[0][1],  # scaled bmi
    children,
    smoker_encoded,
    *region_encoded
]).reshape(1, -1)

# Predict button
if st.button("Predict Insurance Cost"):
    prediction = model.predict(features)[0]
    st.success(f"Predicted Insurance Cost: **${prediction:,.2f}**")
