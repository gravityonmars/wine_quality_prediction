import streamlit as st
import joblib
import pandas as pd 

model = joblib.load('wine_quality_prediction_model.joblib')
scaler = joblib.load('wine_quality_prediction_scaler.joblib')

st.title("Wine Quality Prediction")

features = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol',
]

inputs = [st.number_input(f"Enter {feature}", value=0.0, format="%.5f") for feature in features]

if st.button("Predict"):
    input_df = pd.DataFrame([inputs], columns=features)
    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)[0]

    st.success(f"Predicted Wine Quality: {prediction}")