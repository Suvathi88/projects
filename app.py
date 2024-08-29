import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt

# Title of the app
st.title('Maternal Health Risk Prediction')

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('Maternal Health Risk Data Set.csv')

data = load_data()

# Show the dataset if needed
if st.checkbox("Show Dataset"):
    st.write(data)

# Model Training
st.header("Model Training")

if st.button("Train Model"):
    # Separating features and target
    X = data.drop('RiskLevel', axis=1)
    y = data['RiskLevel']

    # Encode target labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)



    # Building the DNN model
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Input layer
    model.add(Dropout(0.5))  # Dropout for regularization
    model.add(Dense(128, activation='relu'))  # Hidden layer
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))  # Hidden layer
    model.add(Dense(3, activation='softmax'))  # Output layer for 3 classes

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model for 1000 epochs
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=1000, batch_size=32, callbacks=[early_stopping], verbose=2)

    # Save the model
    model.save('maternal_health_dnn_model.h5')

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)

    # Accuracy and classification report
    accuracy = accuracy_score(y_test, y_pred_classes)
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write("Classification Report:\n", classification_report(y_test, y_pred_classes))

    # Plotting the training history
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    st.pyplot(plt)

# Model Prediction
st.header("Predict Risk Level")

# Input fields for prediction
age = st.number_input('Age', min_value=10, max_value=100, value=25)
systolic_bp = st.number_input('Systolic Blood Pressure', min_value=80, max_value=200, value=120)
diastolic_bp = st.number_input('Diastolic Blood Pressure', min_value=40, max_value=120, value=80)
bs = st.number_input('Blood Sugar (mg/dL)', min_value=50, max_value=300, value=100)
body_temp = st.number_input('Body Temperature (°F)', min_value=95.0, max_value=105.0, value=98.6)
heart_rate = st.number_input('Heart Rate (bpm)', min_value=50, max_value=150, value=70)

# Predict button
if st.button("Predict Risk Level"):
    # Load the model
    model = load_model('maternal_health_dnn_model.h5')

    

    # Prepare the input data
    input_data = np.array([[age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate]])
    input_data_scaled = scaler.transform(input_data)

    # Predict using the model
    prediction = model.predict(input_data_scaled)
    predicted_class = np.argmax(prediction, axis=1)
    risk_level = label_encoder.inverse_transform(predicted_class)

    # Output the prediction
    st.write(f"The predicted maternal health risk level is: **{risk_level[0]}**")

