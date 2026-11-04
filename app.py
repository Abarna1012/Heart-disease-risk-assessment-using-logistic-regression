import streamlit as st
import pandas as pd
import joblib
import fitz
import pytesseract
from PIL import Image
import re
from io import BytesIO

st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Risk Assessment")
st.write(
    "Upload medical reports and enter the required clinical details "
    "to assess heart disease risk."
)

# Load existing trained UCI Cleveland model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error("Unable to load model.pkl. Check whether it exists in your GitHub repository.")
    st.stop()


# ---------------- MEDICAL REPORT UPLOAD ----------------

st.header("1. Upload Medical Report")

uploaded_file = st.file_uploader(
    "Upload PDF, JPG or PNG",
    type=["pdf", "jpg", "jpeg", "png"]
)

report_text = ""

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    try:
        if uploaded_file.name.lower().endswith(".pdf"):
            pdf_document = fitz.open(
                stream=file_bytes,
                filetype="pdf"
            )

            for page in pdf_document:
                page_text = page.get_text()

                if page_text.strip():
                    report_text += page_text + "\n"
                else:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    image = Image.open(
                        BytesIO(pix.tobytes("png"))
                    )
                    report_text += pytesseract.image_to_string(image) + "\n"

        else:
            image = Image.open(BytesIO(file_bytes))
            report_text = pytesseract.image_to_string(image)

        st.success("Medical report processed.")

        st.subheader("Extracted Report Text")
        st.text_area(
            "Report content",
            report_text,
            height=200
        )

        # Basic extraction suggestions
        st.subheader("Available Values (Suggestions)")

        patterns = {
            "Age": r"age\s*[:\-]?\s*(\d{1,3})",
            "Total Cholesterol": (
                r"(?:total cholesterol|cholesterol)\s*[:\-]?\s*(\d{2,3})"
            ),
            "Systolic Blood Pressure": (
                r"(?:systolic blood pressure|systolic|sbp)\s*[:\-]?\s*(\d{2,3})"
            ),
            "Fasting Blood Sugar": (
                r"(?:fasting blood sugar|fasting glucose|fbs)\s*[:\-]?\s*(\d{2,3})"
            )
        }

        for label, pattern in patterns.items():
            match = re.search(pattern, report_text, re.IGNORECASE)

            if match:
                st.write(f"**{label}:** {match.group(1)}")
            else:
                st.write(f"**{label}:** Not Available")

        st.warning(
            "Please verify extracted values with the original medical report. "
            "These values are suggestions and are not automatically used for prediction."
        )

    except Exception as e:
        st.error(f"Report processing error: {e}")


# ---------------- PATIENT DETAILS ----------------

st.header("2. Enter Clinical Details")

st.info(
    "Enter values from verified medical records and clinical assessment. "
    "Do not guess missing values."
)

age = st.number_input(
    "Age (years)",
    min_value=1,
    max_value=120,
    value=None,
    placeholder="Enter age"
)

sex = st.selectbox(
    "Sex",
    [None, 0, 1],
    format_func=lambda x: (
        "Select" if x is None
        else "Female" if x == 0
        else "Male"
    )
)

cp = st.selectbox(
    "Chest Pain Type (cp)",
    [None, 0, 1, 2, 3],
    format_func=lambda x: "Select" if x is None else str(x)
)

trestbps = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    min_value=50,
    max_value=250,
    value=None,
    placeholder="Enter resting BP"
)

chol = st.number_input(
    "Total Cholesterol (mg/dl)",
    min_value=50,
    max_value=600,
    value=None,
    placeholder="Enter cholesterol"
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    [None, 0, 1],
    format_func=lambda x: "Select" if x is None else str(x)
)

restecg = st.selectbox(
    "Resting ECG Result (restecg)",
    [None, 0, 1, 2],
    format_func=lambda x: "Select" if x is None else str(x)
)

thalach = st.number_input(
    "Maximum Heart Rate (thalach)",
    min_value=50,
    max_value=250,
    value=None,
    placeholder="Enter maximum exercise heart rate"
)

exang = st.selectbox(
    "Exercise Induced Angina (exang)",
    [None, 0, 1],
    format_func=lambda x: "Select" if x is None else str(x)
)

oldpeak = st.number_input(
    "ST Depression (oldpeak)",
    min_value=0.0,
    max_value=10.0,
    value=None,
    placeholder="Enter oldpeak"
)

slope = st.selectbox(
    "Slope",
    [None, 0, 1, 2],
    format_func=lambda x: "Select" if x is None else str(x)
)

ca = st.selectbox(
    "Number of Major Vessels (ca)",
    [None, 0, 1, 2, 3, 4],
    format_func=lambda x: "Select" if x is None else str(x)
)

thal = st.selectbox(
    "Thalassemia (thal)",
    [None, 0, 1, 2, 3],
    format_func=lambda x: "Select" if x is None else str(x)
)


# ---------------- PREDICTION ----------------

st.header("3. Risk Assessment")

verified = st.checkbox(
    "I have verified all 13 clinical values with the original records."
)

if st.button("Assess Heart Disease Risk"):

    values = [
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak,
        slope, ca, thal
    ]

    if any(value is None for value in values):
        st.error(
            "Some clinical values are missing. "
            "Please enter all 13 values before prediction."
        )

    elif not verified:
        st.warning("Please verify all clinical values before prediction.")

    else:
        input_data = pd.DataFrame(
            [[
                age, sex, cp, trestbps, chol, fbs,
                restecg, thalach, exang, oldpeak,
                slope, ca, thal
            ]],
            columns=[
                "age", "sex", "cp", "trestbps", "chol",
                "fbs", "restecg", "thalach", "exang",
                "oldpeak", "slope", "ca", "thal"
            ]
        )

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        if prediction == 1:
            risk = probability[1] * 100
            st.error(
                f"Higher Risk of Heart Disease: {risk:.2f}%"
            )
        else:
            risk = probability[0] * 100
            st.success(
                f"Lower Risk of Heart Disease: {risk:.2f}%"
            )

        st.caption(
            "This is a machine-learning screening result, "
            "not a medical diagnosis. Consult a qualified doctor."
        )


# ---------------- MODEL PERFORMANCE ----------------

st.header("4. Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", "86.89%")

with col2:
    st.metric("ROC-AUC", "95.35%")


# ---------------- HARDWARE STATUS ----------------

st.header("5. Planned Hardware Integration")

st.write("Hardware integration is planned for the next development stage.")

st.write("- ESP32: Wi-Fi communication")
st.write("- MAX30102: Heart rate and SpO2")
st.write("- AD8232: ECG signal acquisition")

st.caption(
    "Sensor readings will be displayed separately. "
    "They will not be substituted for the 13 clinical model inputs."
)
