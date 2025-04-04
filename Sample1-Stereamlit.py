import streamlit as st
import pandas as pd
import pickle

# Load the model and scaler
with open('pkl/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('pk1/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit app
st.title("Loan Approval Prediction")
st.write("Enter the applicant details to predict loan approval status.")

# Input fields
education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
income_annum = st.number_input("Annual Income (in INR)", min_value=0.0, value=500000.0)
loan_amount = st.number_input("Loan Amount (in INR)", min_value=0.0, value=200000.0)
loan_term = st.number_input("Loan Term (in years)", min_value=1, max_value=30, value=10)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750)
residential_assets_value = st.number_input("Residential Assets Value (in INR)", min_value=0.0, value=300000.0)
commercial_assets_value = st.number_input("Commercial Assets Value (in INR)", min_value=0.0, value=150000.0)
luxury_assets_value = st.number_input("Luxury Assets Value (in INR)", min_value=0.0, value=200000.0)
bank_asset_value = st.number_input("Bank Asset Value (in INR)", min_value=0.0, value=100000.0)

# Predict button
if st.button("Predict"):
    # Prepare input data
    input_data = pd.DataFrame({
        'education': [1 if education == 'Graduate' else 0],
        'self_employed': [1 if self_employed == 'Yes' else 0],
        'income_annum': [income_annum],
        'loan_amount': [loan_amount],
        'loan_term': [loan_term],
        'cibil_score': [cibil_score],
        'residential_assets_value': [residential_assets_value],
        'commercial_assets_value': [commercial_assets_value],
        'luxury_assets_value': [luxury_assets_value],
        'bank_asset_value': [bank_asset_value]
    })
    
    # Scale numerical features
    numerical_cols = ['income_annum', 'loan_amount', 'loan_term', 'cibil_score', 
                      'residential_assets_value', 'commercial_assets_value', 
                      'luxury_assets_value', 'bank_asset_value']
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    result = "Approved" if prediction == 1 else "Not Approved"
    
    # Display result
    st.success(f"Loan Status: {result}")

# Add some styling
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)
