import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the model
model = load_model('maternal_health_dnn_model.h5')

# Load the scaler and label encoder
scaler = StandardScaler()  # Assuming scaler is already fitted earlier
# Label encoder would also need to be loaded or recreated similarly
label_encoder = LabelEncoder()  # Assuming label encoder is already fitted

# Streamlit app
st.title('Maternal Health Risk Prediction')

# Input fields
age = st.number_input('Age', min_value=10, max_value=100, value=25)
systolic_bp = st.number_input('Systolic Blood Pressure', min_value=80, max_value=200, value=120)
diastolic_bp = st.number_input('Diastolic Blood Pressure', min_value=40, max_value=120, value=80)
bs = st.number_input('Blood Sugar (mg/dL)', min_value=50, max_value=300, value=100)
body_temp = st.number_input('Body Temperature (°F)', min_value=95.0, max_value=105.0, value=98.6)
heart_rate = st.number_input('Heart Rate (bpm)', min_value=50, max_value=150, value=70)

# Button to make predictions
if st.button('Predict Risk Level'):
    # Prepare the input data
    input_data = np.array([[age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate]])
    input_data_scaled = scaler.transform(input_data)

    # Predict using the model
    prediction = model.predict(input_data_scaled)
    predicted_class = np.argmax(prediction, axis=1)
    risk_level = label_encoder.inverse_transform(predicted_class)

    # Output the prediction
    st.write(f'The predicted maternal health risk level is: **{risk_level[0]}**')

