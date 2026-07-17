import streamlit as st
import pandas as pd
import pickle
import os

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(page_title="Student Performance Dashboard", page_icon="📈", layout="wide")

# 2. Load the trained KNeighborsRegressor model
@st.cache_resource
def load_model():
    # Check for either filename just in case
    if os.path.exists('model (1).pkl'):
        model_path = 'model (1).pkl'
    elif os.path.exists('model.pkl'):
        model_path = 'model.pkl'
    else:
        return None

    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# 3. Sidebar for Inputs
st.sidebar.header("⚙️ Input Parameters")
st.sidebar.markdown("Adjust the student's metrics below:")

hours_studied = st.sidebar.number_input("Hours Studied (per day)", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
sleep_hours = st.sidebar.number_input("Sleep Hours (per night)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
attendance_percent = st.sidebar.slider("Attendance Percentage (%)", min_value=0, max_value=100, value=85)
previous_scores = st.sidebar.number_input("Previous Exam Score", min_value=0.0, max_value=100.0, value=75.0, step=1.0)

st.sidebar.markdown("---")
predict_button = st.sidebar.button("Generate Prediction", type="primary", use_container_width=True)

# 4. Main Dashboard UI
st.title("📈 Student Performance Dashboard")
st.markdown("This dashboard predicts a student's final exam score based on their study habits, attendance, and previous academic history.")

if model is None:
    st.error("🚨 **Model not found!** Please ensure `model.pkl` or `model (1).pkl` is in the same folder as this script.")
else:
    # 5. Prediction Logic
    if predict_button:
        with st.spinner("Analyzing data..."):
            # Construct the input DataFrame matching the exact feature names
            # 'exam_score' is included as a dummy variable (0.0) to bypass the training error[cite: 2]
            input_data = pd.DataFrame({
                'hours_studied': [hours_studied],
                'sleep_hours': [sleep_hours],
                'attendance_percent': [attendance_percent],
                'previous_scores': [previous_scores],
                'exam_score': [0.0]  
            })
            
            try:
                # Generate prediction
                prediction = model.predict(input_data)
                predicted_score = float(prediction[0])
                
                # Calculate the difference from the previous score
                score_delta = predicted_score - previous_scores
                
                st.markdown("### Prediction Results")
                st.divider()
                
                # Display metrics in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Predicted Exam Score", 
                        value=f"{predicted_score:.1f}%", 
                        delta=f"{score_delta:.1f}% vs Previous",
                        delta_color="normal"
                    )
                with col2:
                    st.metric(label="Attendance", value=f"{attendance_percent}%")
                with col3:
                    st.metric(label="Daily Study Hours", value=f"{hours_studied} hrs")
                
                # Visual Progress Bar for the score
                st.write("")
                st.write("**Score Projection Gauge:**")
                # Ensure the progress bar doesn't break if the prediction goes slightly over 100
                st.progress(min(int(predicted_score), 100))
                
                if predicted_score >= 80:
                    st.success("✨ Excellent trajectory! The student is on track for a high grade.")
                elif predicted_score >= 60:
                    st.info("👍 Solid performance, but there is still room for improvement.")
                else:
                    st.warning("⚠️ The student is at risk of a low score. Consider increasing study hours or attendance.")
                
            except Exception as e:
                st.error(f"Prediction Error: {e}")
    else:
        st.info("👈 Adjust the parameters in the sidebar and click **Generate Prediction** to see the results.")
