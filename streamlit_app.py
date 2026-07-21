
import streamlit as st
import pandas as pd
import pickle

# Page configuration
st.set_page_config(
    page_title="Healthcare AI Project",
    page_icon="🏥",
    layout="wide"
)

# Load trained Random Forest model

with open("random_forest_standard.pkl", "rb") as f:
    model = pickle.load(f)

# Title
st.title("🏥 Healthcare Synthetic Data Project")

st.write(
    "Synthetic Medical Diagnosis Record Generation Using CTGAN"
)

st.divider()

# Project overview
st.header("📌 Project Overview")

st.write("""
This application demonstrates the generation of synthetic healthcare
data using CTGAN and diabetes prediction using a Random Forest
Machine Learning model.
""")

# Sidebar
st.sidebar.header("Patient Information")

gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male", "Other"]
)

age = st.sidebar.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=40.0
)

hypertension = st.sidebar.selectbox(
    "Hypertension",
    [0, 1]
)

heart_disease = st.sidebar.selectbox(
    "Heart Disease",
    [0, 1]
)

smoking_history = st.sidebar.selectbox(
    "Smoking History",
    [
        "No Info",
        "current",
        "ever",
        "former",
        "never",
        "not current"
    ]
)

bmi = st.sidebar.number_input(
    "BMI",
    min_value=10.0,
    max_value=70.0,
    value=25.0
)

hba1c_level = st.sidebar.number_input(
    "HbA1c Level",
    min_value=3.0,
    max_value=10.0,
    value=5.5
)

blood_glucose_level = st.sidebar.number_input(
    "Blood Glucose Level",
    min_value=50.0,
    max_value=400.0,
    value=120.0
)

# Encoding categorical values
gender_mapping = {
    "Female": 0,
    "Male": 1,
    "Other": 2
}

smoking_mapping = {
    "No Info": 0,
    "current": 1,
    "ever": 2,
    "former": 3,
    "never": 4,
    "not current": 5
}

gender_encoded = gender_mapping[gender]

smoking_encoded = smoking_mapping[smoking_history]

# Prediction section
st.header("🔍 Diabetes Prediction")

if st.button("Predict Diabetes Risk"):

    input_data = pd.DataFrame({
        "gender": [gender_encoded],
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "smoking_history": [smoking_encoded],
        "bmi": [bmi],
        "HbA1c_level": [hba1c_level],
        "blood_glucose_level": [blood_glucose_level]
    })

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    diabetes_probability = probability[0][1] * 100

    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.error(
            "⚠️ Model Prediction: Diabetes Risk Detected"
        )

    else:

        st.success(
            "✅ Model Prediction: No Diabetes Risk Detected"
        )

    st.metric(
        "Predicted Diabetes Probability",
        f"{diabetes_probability:.2f}%"
    )

    st.warning(
        "This application is for educational and research purposes only "
        "and should not be used as a medical diagnosis."
    )
