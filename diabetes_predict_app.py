
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# Load the model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("model_rf.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

st.title("Diabetes Prediction App")
st.write("Enter the patient data to check if they are likely diabetic.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
bp = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=10, max_value=100, value=30)

input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])

# Predict button
if st.button("Predict"):
    # Replace 0s in specific fields if needed (simulate preprocessing logic)
    input_df = pd.DataFrame(input_data, columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                                                  "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])
    imputer = SimpleImputer(strategy="median")
    input_df = pd.DataFrame(imputer.fit_transform(input_df), columns=input_df.columns)
    input_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]
    
    if prediction == 1:
        st.error(f"The person is likely diabetic. (Confidence: {prob:.2f})")
    else:
        st.success(f"The person is unlikely to be diabetic. (Confidence: {1 - prob:.2f})")
