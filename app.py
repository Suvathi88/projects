import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model
model = load_model('maternal_health_dnn_model.h5')

# Define the scaler and label encoder as they were used during training
# Note: Replace these with the actual values used during training if possible

# Example values used during training (You should replace these with actual values)
scaler = StandardScaler()
scaler.mean_ = np.array([50, 130, 85, 7.0, 99.0, 75])  # Replace with actual means
scaler.scale_ = np.array([20, 30, 20, 3.0, 5.0, 10])   # Replace with actual scales

label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['low', 'medium', 'high'])  # Replace with actual classes

# Streamlit app
st.title("Maternal Health Risk Prediction")

# Input fields for user data
age = st.number_input("Age", min_value=10, max_value=100, value=30)
systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=50, max_value=200, value=120)
diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=30, max_value=130, value=80)
bs = st.number_input("Blood Sugar Level (mmol/L)", min_value=1.0, max_value=20.0, value=6.5)
body_temp = st.number_input("Body Temperature (°F)", min_value=90.0, max_value=110.0, value=98.6)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=180, value=72)

# Predict function
def predict_risk_level(age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate):
    # Prepare the input data
    new_data = np.array([[age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate]])
    
    # Scale the data
    new_data_scaled = scaler.transform(new_data)
    
    # Predict the risk level
    prediction = model.predict(new_data_scaled)
    predicted_class = prediction.argmax(axis=1)
    
    # Decode the predicted class back to the original label
    predicted_risk_level = label_encoder.inverse_transform(predicted_class)
    
    return predicted_risk_level[0]

# Predict button
if st.button("Predict"):
    risk_level = predict_risk_level(age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate)
    st.write(f"Predicted Risk Level: **{risk_level}**")



