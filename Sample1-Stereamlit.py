import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Simulate dataset (replace with your actual dataset if available)
data = pd.DataFrame({
    'education': np.random.choice(['Graduate', 'Not Graduate'], 1000),
    'self_employed': np.random.choice(['Yes', 'No'], 1000),
    'income_annum': np.random.normal(500000, 150000, 1000),
    'loan_amount': np.random.normal(200000, 75000, 1000),
    'loan_term': np.random.randint(1, 20, 1000),
    'cibil_score': np.random.randint(300, 900, 1000),
    'residential_assets_value': np.random.normal(300000, 100000, 1000),
    'commercial_assets_value': np.random.normal(150000, 75000, 1000),
    'luxury_assets_value': np.random.normal(200000, 80000, 1000),
    'bank_asset_value': np.random.normal(100000, 50000, 1000),
    'loan_status': np.random.choice([0, 1], 1000)
})

# If you have your own dataset, uncomment and modify this line:
# data = pd.read_csv('path_to_your_dataset.csv')

# Preprocessing
def preprocess_data(df):
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    le = LabelEncoder()
    for col in ['education', 'self_employed']:
        df[col] = le.fit_transform(df[col])
    
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    scaler = StandardScaler()
    numerical_cols = ['income_annum', 'loan_amount', 'loan_term', 'cibil_score', 
                      'residential_assets_value', 'commercial_assets_value', 
                      'luxury_assets_value', 'bank_asset_value']
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    return X, y, scaler

X, y, scaler = preprocess_data(data)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest with hyperparameter tuning
param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

#TODO with Decision Tree

# Evaluate
y_pred = best_rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1]):.4f}")
print(classification_report(y_test, y_pred))

import os

# Set page configuration for a professional look
st.set_page_config(page_title="Bank Loan Approval Prediction", layout="wide", initial_sidebar_state="expanded")

# Load model and scaler with error handling
# try:
#     if not os.path.exists('best_model.pkl') or not os.path.exists('scaler.pkl'):
#         st.error("Model or scaler files not found! Please run 'train_model.py' first.")
#         st.stop()
#     with open('best_model.pkl', 'rb') as f:
#         model = pickle.load(f)
#     with open('scaler.pkl', 'rb') as f:
#         scaler = pickle.load(f)
# except Exception as e:
#     st.error(f"Error loading files: {str(e)}")
#     st.stop()

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
        prediction = best_rf.predict(input_data)[0]
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