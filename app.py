import streamlit as st
import pickle
import numpy as np

# Load the model
with open('heart_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Set page title
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# Title and description
st.title("💓 Heart Disease Prediction System")
st.markdown("Enter the following information to predict the likelihood of heart disease:")

# User input form
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])

    with col2:
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
        chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=240)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True; 0=False)", [0, 1])

    with col3:
        restecg = st.selectbox("Resting ECG results (0,1,2)", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina (1=Yes; 0=No)", [0, 1])

    # More fields
    oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.0, 1.0, 0.1)
    slope = st.selectbox("Slope of peak exercise ST segment", [0, 1, 2])
    ca = st.selectbox("Number of major vessels colored by fluoroscopy (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thal (0=Normal; 1=Fixed Defect; 2=Reversible Defect)", [0, 1, 2])

    submit = st.form_submit_button("Predict")

# Prediction
if submit:
    # Convert sex to binary
    sex_bin = 1 if sex == "Male" else 0

    features = np.array([[age, sex_bin, cp, trestbps, chol, fbs,
                          restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠️ The model predicts a high likelihood of heart disease.")
    else:
        st.success("✅ The model predicts a low likelihood of heart disease.")
