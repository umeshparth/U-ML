import streamlit as st
import pandas as pd
import os
import pickle

# Set page configuration for a professional look
st.set_page_config(page_title="Bank Loan Approval Prediction", layout="wide", initial_sidebar_state="expanded")

# Load model and scaler with error handling
try:
    if not os.path.exists('best_model.pkl') or not os.path.exists('scaler.pkl'):
        st.error("Model or scaler files not found! Please run 'train_model.py' first.")
        st.stop()
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"Error loading files: {str(e)}")
    st.stop()

# Custom CSS for professional styling
st.markdown("""
    <style>
    /* General styling */
    body {
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        background-color: #f5f6f5;
    }
    .title {
        font-size: 32px;
        color: #362479; /* Bank-like navy blue */
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        color: #555555;
        text-align: center;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #362479;
        color: white;
        border-radius: 8px;
        font-size: 16px;
        padding: 10px 20px;
        width: 100%;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #4f37a6; /* Lighter blue on hover */
    }
    .stTextInput, .stNumberInput, .stSelectbox {
        border: 1px solid #cccccc;
        border-radius: 5px;
        padding: 5px;
    }
    .stSuccess {
        background-color: #e6f3ff;
        border: 1px solid #362479;
        border-radius: 5px;
        padding: 10px;
        font-size: 18px;
        color: #362479;
    }
    .section-header {
        font-size: 22px;
        color: #362479;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">Bank Loan Approval Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Assess loan applications with precision and efficiency</div>', unsafe_allow_html=True)

# Sidebar for instructions or branding
with st.sidebar:
    st.image("UVRBank.png")  # Replace with actual bank logo URL
    st.write("### Instructions")
    st.write("Enter the applicant’s details below to predict loan approval status. Ensure all fields are filled accurately.")
    st.write("---")
    st.write("**Contact Support**: support@bankname.com")

# Main content with containers
with st.container():
    st.markdown('<div class="section-header">Applicant Information</div>', unsafe_allow_html=True)
    
    # Two-column layout with grouped inputs
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():  # Group 1: Personal Details
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.write("#### Personal Details")
            name = st.text_input("Applicant Name", value="John Doe", help="Enter applicant’s full name")
            contact_number = st.text_input("Contact Number", value="9876543210", 
                                           help="Enter applicant’s 10-digit phone number")
            education = st.selectbox("Education Level", ['Graduate', 'Not Graduate'], 
                                     help="Applicant’s education status")
            self_employed = st.selectbox("Employment Status", ['Yes', 'No'], 
                                         help="Is the applicant self-employed?")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with st.container():  # Group 2: Loan Details
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.write("#### Loan Details")
            loan_amount = st.number_input("Loan Amount (INR)", min_value=0.0, value=200000.0, step=10000.0, 
                                          format="%.2f", help="Requested loan amount")
            loan_term = st.number_input("Loan Term (Years)", min_value=1, max_value=30, value=10, 
                                        help="Loan duration in years")
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        with st.container():  # Group 3: Financial Details
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.write("#### Financial Details")
            income_annum = st.number_input("Annual Income (INR)", min_value=0.0, value=500000.0, step=10000.0, 
                                           format="%.2f", help="Applicant’s annual income")
            cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750, 
                                          help="Applicant’s credit score (300-900)")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with st.container():  # Group 4: Asset Details
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.write("#### Asset Details")
            residential_assets_value = st.number_input("Residential Assets (INR)", min_value=0.0, value=300000.0, 
                                                       step=10000.0, format="%.2f", help="Value of residential assets")
            commercial_assets_value = st.number_input("Commercial Assets (INR)", min_value=0.0, value=150000.0, 
                                                      step=10000.0, format="%.2f", help="Value of commercial assets")
            luxury_assets_value = st.number_input("Luxury Assets (INR)", min_value=0.0, value=200000.0, 
                                                  step=10000.0, format="%.2f", help="Value of luxury assets")
            bank_asset_value = st.number_input("Bank Assets (INR)", min_value=0.0, value=100000.0, 
                                               step=10000.0, format="%.2f", help="Value of bank-held assets")
            st.markdown('</div>', unsafe_allow_html=True)

    # Predict button
    if st.button("Predict Loan Approval"):
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
        st.markdown(f'<div class="stSuccess">Loan Status: <b>{result}</b></div>', unsafe_allow_html=True)
        if prediction == 1:
            st.write("**Note**: Loan approved subject to final verification.")
        else:
            st.write("**Note**: Review applicant details for potential improvements.")

# Footer
st.markdown("---")
st.write("© 2025 Bank Name | Powered by Advanced ML Technology")