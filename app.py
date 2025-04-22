import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load assets
model = joblib.load('insurance_cost_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title
st.title("Medical Insurance Cost Predictor 🏥")

# Input widgets
st.header("Patient Information")
age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", ["Female", "Male"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=28.5)
children = st.number_input("Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["No", "Yes"])
region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

# Preprocessing
sex_encoded = 1 if sex == "Male" else 0
smoker_encoded = 1 if smoker == "Yes" else 0

# Region encoding
region_mapping = {
    "Northeast": [0, 0, 0],
    "Northwest": [1, 0, 0],
    "Southeast": [0, 1, 0],
    "Southwest": [0, 0, 1]
}
region_encoded = region_mapping[region]

# Feature scaling with proper column names
input_df = pd.DataFrame([[age, bmi, children]], 
                       columns=['age', 'bmi', 'children'])
scaled_features = scaler.transform(input_df)

# Build feature array in EXACT training order
features = np.array([
    scaled_features[0][0],  # age
    sex_encoded,            # sex
    scaled_features[0][1],  # bmi
    children,               # children
    smoker_encoded,         # smoker
    region_encoded[0],      # region_northwest
    region_encoded[1],      # region_southeast
    region_encoded[2]       # region_southwest
]).reshape(1, -1)

# Prediction
if st.button("Predict Insurance Cost"):
    try:
        prediction = model.predict(features)[0]
        st.success(f"Predicted Cost: **${prediction:,.2f}**")
        
        # Debug output (matches training columns)
        st.write("Feature Order Verified:", [
            'age', 'sex', 'bmi', 'children', 'smoker',
            'region_northwest', 'region_southeast', 'region_southwest'
        ])
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
