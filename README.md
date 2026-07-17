# 🎓 Student Performance Dashboard

A machine learning web application built with Streamlit that predicts a student's final exam score based on their study habits, attendance, and historical academic performance. 

**Live Demo:** [Student Performance Dashboard](https://studentscorepredictb12-rtbma4nvxsjrmsaqsckb5e.streamlit.app/)

## 🌟 Features
* **Interactive UI:** A clean, easy-to-use sidebar for adjusting input parameters.
* **Real-Time Predictions:** Instantly calculates the predicted exam score using a K-Nearest Neighbors (KNN) regression model.
* **Performance Metrics:** Visually compares the predicted score against previous exam scores to show trajectory (improvement or decline).
* **Visual Progress Indicators:** Dynamic gauges that provide immediate, color-coded feedback on the student's projected outcome.

## 🛠️ Tech Stack
* **Python**
* **Streamlit** (Web framework)
* **Scikit-learn** (Machine learning)
* **Pandas & NumPy** (Data manipulation)

## 🚀 How to Run Locally

1. **Download the project files** to your local machine.
2. **Ensure you have the required files** in the same directory:
   * `app.py`
   * `model.pkl` (The trained KNN model expecting 4 input features)
   * `requirements.txt`
3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
