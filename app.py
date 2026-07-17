import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Configure the page layout
st.set_page_config(page_title="Exam Score Predictor", page_icon="🎓", layout="centered")

# Load the trained KNeighborsRegressor model
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Error: 'model.pkl' not found. Please ensure the file is in the same directory as app.py.")
        return None

model = load_model()

# App Header
st.title("🎓 Student Exam Score Predictor")
st.markdown("---")
st.write("Enter the student's academic and lifestyle details below to predict their final exam score.")

# Create a clean two-column layout for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("Study Metrics")
    hours_studied = st.number_input("Hours Studied (per day)", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
    attendance_percent = st.slider("Attendance Percentage (%)", min_value=0, max_value=100, value=85)

with col2:
    st.subheader("Historical & Health Metrics")
    previous_scores = st.number_input("Previous Exam Score", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
    sleep_hours = st.number_input("Sleep Hours (per night)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)

st.markdown("---")

# Prediction logic
if st.button("Predict Exam Score", type="primary", use_container_width=True):
    if model is not None:
        # Construct the input DataFrame matching the exact feature names from the pickle file
        input_data = pd.DataFrame({
            'hours_studied': [hours_studied],
            'sleep_hours': [sleep_hours],
            'attendance_percent': [attendance_percent],
            'previous_scores': [previous_scores]
        })
        
        try:
            # Generate prediction
            prediction = model.predict(input_data)
            predicted_score = prediction[0]
            
            # Display result
            st.success(f"### Predicted Exam Score: **{predicted_score:.2f}**")
            
        except ValueError as e:
            # Fallback in case 'exam_score' was accidentally included as a training feature
            st.error(f"Prediction Error: {e}")
            if "expecting 5 features" in str(e).lower():
                st.warning("⚠️ It looks like the model was trained with the target variable ('exam_score') as one of the input features. You will need to retrain the `model.pkl` file, ensuring 'exam_score' is dropped from the X (training) dataset.")
