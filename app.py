import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")

# Page settings
st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Risk Assessment")
st.write("Enter the patient's details to assess the heart disease risk.")

# User inputs
age = st.number_input("Age", min_value=1, max_value=120, value=50)

sex = st.selectbox(
    "Sex", [0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

cp = st.number_input("Chest Pain Type (cp)", min_value=0, max_value=3, value=0)

trestbps = st.number_input(
    "Resting Blood Pressure", min_value=50, max_value=250, value=120
)

chol = st.number_input(
    "Cholesterol", min_value=50, max_value=600, value=200
)

fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

restecg = st.number_input(
    "Resting ECG (restecg)", min_value=0, max_value=2, value=0
)

thalach = st.number_input(
    "Maximum Heart Rate (thalach)", min_value=50, max_value=250, value=150
)

exang = st.selectbox("Exercise Induced Angina", [0, 1])

oldpeak = st.number_input(
    "ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0
)

slope = st.number_input("Slope", min_value=0, max_value=2, value=1)

ca = st.number_input(
    "Number of Major Vessels (ca)", min_value=0, max_value=4, value=0
)

thal = st.number_input("Thal", min_value=0, max_value=3, value=1)

# Prediction button
if st.button("Assess Heart Disease Risk"):

    input_data = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak,
        slope, ca, thal
    ]], columns=[
        'age', 'sex', 'cp', 'trestbps', 'chol',
        'fbs', 'restecg', 'thalach', 'exang',
        'oldpeak', 'slope', 'ca', 'thal'
    ])

    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        risk = probability[1] * 100
        st.error(f"⚠️ Higher risk of heart disease = {risk:.2f}%")
    else:
        risk = probability[0] * 100
        st.success(f"✅ Lower risk of heart disease = {risk:.2f}%")

# Model Performance
st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", "86.89%")

with col2:
    st.metric("ROC-AUC", "95.35%")
