import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and scaler with error handling
try:
    model = joblib.load('insurance_cost_model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Title and inputs
st.title("Medical Insurance Cost Predictor 🏥")

# Input widgets with default values matching training data distribution
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=39)
    sex = st.selectbox("Sex", ["Female", "Male"], index=0)
    bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=30.66)
    children = st.number_input("Children", min_value=0, max_value=10, value=1)
    smoker = st.selectbox("Smoker", ["No", "Yes"], index=0)
    region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"], index=2)
    
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        # Preprocessing
        sex_encoded = 1 if sex == "Male" else 0
        smoker_encoded = 1 if smoker == "Yes" else 0
        
        # Region encoding (must match training order)
        region_mapping = {
            "Northeast": [0, 0, 0],
            "Northwest": [1, 0, 0],
            "Southeast": [0, 1, 0],
            "Southwest": [0, 0, 1]
        }
        region_encoded = region_mapping[region]

        # Create DataFrame for scaling with correct column names
        scaling_df = pd.DataFrame([[age, bmi, children]], 
                                columns=['age', 'bmi', 'children'])
        
        # Scale features
        scaled_features = scaler.transform(scaling_df)

        # Create final feature array in EXACT training order
        features = np.array([
            scaled_features[0][0],  # age
            sex_encoded,            # sex
            scaled_features[0][1],  # bmi
            children,               # children
            smoker_encoded,         # smoker
            *region_encoded         # region features
        ]).reshape(1, -1)

        # Create DataFrame with proper column names for the model
        feature_columns = ['age', 'sex', 'bmi', 'children', 'smoker',
                         'region_northwest', 'region_southeast', 'region_southwest']
        prediction_df = pd.DataFrame(features, columns=feature_columns)

        # Make prediction
        prediction = model.predict(prediction_df)[0]
        st.success(f"Predicted Insurance Cost: **${prediction:,.2f}**")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")