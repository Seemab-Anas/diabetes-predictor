import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="⚕️")

# Load the trained model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title of the app
st.title("Diabetes Risk Prediction")

# Sidebar - User Input
st.sidebar.header("User Input")
pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.sidebar.slider("Glucose Level", min_value=0, max_value=200, value=100)
blood_pressure = st.sidebar.slider("Blood Pressure", min_value=0, max_value=150, value=70)
skin_thickness = st.sidebar.slider("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.sidebar.slider("Insulin", min_value=0, max_value=300, value=79)
bmi = st.sidebar.slider("BMI", min_value=0.0, max_value=60.0, value=25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.sidebar.slider("Age", min_value=10, max_value=100, value=30)

# Prepare input data
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

# Standardize the input data using the saved scaler
input_data = scaler.transform(input_data)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)  # Make prediction

    if prediction[0] == 1:
        st.error("🚨 High Risk of Diabetes! Consult a doctor.")
    else:
        st.success("✅ Low Risk of Diabetes. Stay healthy!")


# Footer with my info
st.markdown(
    """
    <div style="position: fixed; bottom: 10px; right: 10px; text-align: right; font-size: 14px; color: gray;">
        <b>👨‍💻 Developed by <span style="color: orange;">Seemab Anas</span></b><br>
        📧 <a href="mailto:seemabanas2022@gmail.com" style="color: gray;">seemabanas2022@gmail.com</a><br>
        🔗 <a href="https://www.linkedin.com/in/seemabanas/" target="_blank" style="color: gray;">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True
)

